import torch
from torchvision import datasets, transforms
from torch.autograd import gradcheck
from torch.utils.data import DataLoader
from torchviz import make_dot
from tqdm import tqdm


class IterativeFixedPoint(torch.nn.Module):

    def __init__(self, tol, max_iter, features):
        super().__init__()
        self.tol = tol
        self.projection = torch.nn.Linear(in_features=features,
                                          out_features=features)
        self.max_iter = max_iter

    def forward(self, x):
        z = torch.zeros_like(x)
        for iter in range(self.max_iter):
            z_new = torch.tanh(self.projection(z) + x)
            if torch.linalg.vector_norm(z - z_new) < self.tol:
                print(f'Breaking after {1 + iter} iterations')
                return z_new
            z = z_new
        return z


class NewtonFixedPoint(torch.nn.Module):

    def __init__(self, tol, max_iter, features):
        super().__init__()
        self.tol = tol
        self.projection = torch.nn.Linear(in_features=features,
                                          out_features=features)
        self.max_iter = max_iter
        self.iterations = 0

    def forward(self, x):
        z = torch.tanh(x)
        self.iterations = 0
        for iter in range(self.max_iter):
            self.iterations += 1
            z_linear = self.projection(z) + x
            dist = torch.linalg.vector_norm(z - torch.tanh(z_linear))
            if dist < self.tol:
                break
            # Newton step
            z_diff = 1.0 / torch.cosh(z_linear)**2
            J = torch.eye(n=z_diff.size(1))[
                None, :, :] - z_diff[:, :,
                                     None] * self.projection.weight[None, :, :]
            newton_step = torch.linalg.solve(J, z - torch.tanh(z_linear))
            z = z - newton_step

        return z


class NewtonFixedPointImplicitGrad(torch.nn.Module):

    def __init__(self, tol, max_iter, features):
        super().__init__()
        self.tol = tol
        self.projection = torch.nn.Linear(in_features=features,
                                          out_features=features)
        self.max_iter = max_iter
        self.iterations = 0

    def forward(self, x):
        with torch.no_grad():
            z = torch.tanh(x)
            self.iterations = 0
            for iter in range(self.max_iter):
                self.iterations += 1
                z_linear = self.projection(z) + x
                dist = torch.linalg.vector_norm(z - torch.tanh(z_linear))
                if dist < self.tol:
                    break
                # Newton step
                z_diff = 1.0 / torch.cosh(z_linear)**2
                J = torch.eye(n=z_diff.size(1))[
                    None, :, :] - z_diff[:, :, None] * self.projection.weight[
                        None, :, :]
                newton_step = torch.linalg.solve(J, z - torch.tanh(z_linear))
                z = z - newton_step
        # reengage autograd and add the gradient hook
        z = torch.tanh(self.projection(z) + x)

        def manipulate_grad(grad):
            result = torch.linalg.solve(J.transpose(1, 2), grad[:, :, None])
            return result[:, :, 0]

        z.register_hook(lambda grad: manipulate_grad(grad))
        return z


class AndersonFixedPoint(torch.nn.Module):

    def __init__(self, tol, lookback, max_iter, features):
        super().__init__()
        self.tol = tol
        self.projection = torch.nn.Linear(in_features=features,
                                          out_features=features)
        self.max_iter = max_iter
        self.lookback = lookback
        self.iterations = 0

    def step(self, z, x):
        return torch.tanh(self.projection(z) + x)

    def forward(self, x):
        with torch.no_grad():
            z = torch.tanh(x)
            X_lookback = torch.zeros(self.lookback, z.shape[0], z.shape[1])
            G_lookback = torch.zeros(self.lookback, z.shape[0], z.shape[1])
            for iter in range(self.max_iter):
                self.iterations += 1
                # Save stuff for computing gradients later
                z_linear = self.projection(z) + x
                z_diff = 1.0 / torch.cosh(z_linear)**2
                J = torch.eye(n=z_diff.size(1))[
                    None, :, :] - z_diff[:, :, None] * self.projection.weight[
                        None, :, :]
                z_new = torch.tanh(z_linear)
                if torch.linalg.vector_norm(z - z_new) < self.tol:
                    break
                m_k = min(self.lookback, iter)
                X_lookback = torch.roll(X_lookback, shifts=1, dims=0)
                X_lookback[0] = z
                G_lookback = torch.roll(G_lookback, shifts=1, dims=0)
                G_lookback[0] = z_new - z
                if m_k > 10:  # Less noisy estimate ?
                    LR_X = G_lookback[1:(m_k + 1)] - G_lookback[0]
                    A = torch.einsum('mbd, nbd-> bmn', LR_X, LR_X)
                    B = torch.einsum('lbd, bd -> bl', LR_X, G_lookback[0])
                    gamma = torch.linalg.solve(A, B)
                    matrix = (X_lookback[1:(m_k + 1)] - X_lookback[0] + LR_X)
                    z_new = z_new + torch.einsum('lbd,bl->bd', matrix, gamma)
                z = z_new
        # reengage autograd and add the gradient hook
        z = torch.tanh(self.projection(z) + x)

        def manipulate_grad(grad):
            result = torch.linalg.solve(J.transpose(1, 2), grad[:, :, None])
            return result[:, :, 0]

        z.register_hook(lambda grad: manipulate_grad(grad))
        return z


# fp = IterativeFixedPoint(1e-5, 10000, 50)
# fp = NewtonFixedPoint(1e-5, 10000, 50)
fp = AndersonFixedPoint(1e-5, 15, 10000, 50)
x = torch.randn(10, 50)
z = fp(x)
# make_dot(z,
#          params=dict(list(fp.named_parameters()))).render("NewtonFixedPoint",
#                                                           format="png")
print(torch.linalg.vector_norm(z - fp.step(z, x)))
#exit(0)

# Execute a gradcheck
#layer = NewtonFixedPointImplicitGrad(1e-5, 100, 5).double()
layer = AndersonFixedPoint(1e-5, 15, 100, 5).double()
assert gradcheck(layer,
                 torch.randn(3, 5, requires_grad=True, dtype=torch.double),
                 check_undefined_grad=False)

mnist_train = datasets.MNIST(".",
                             train=True,
                             download=True,
                             transform=transforms.ToTensor())
mnist_test = datasets.MNIST(".",
                            train=False,
                            download=True,
                            transform=transforms.ToTensor())
train_dataloader = DataLoader(mnist_train, batch_size=100, shuffle=True)
test_dataloader = DataLoader(mnist_test, batch_size=100, shuffle=True)
torch.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else "cpu"
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 100),
    #NewtonFixedPointImplicitGrad(1e-4, 40, 100),
    AndersonFixedPoint(1e-4, 15, 40, 100),
    torch.nn.Linear(100, 10)).to(device)
optim = torch.optim.SGD(model.parameters(), lr=1e-1)


def epoch(loader, model, opt=None, monitor=None):
    total_loss = 0.0
    total_err = 0.0
    total_monitor = 0.0
    for X, y in tqdm(loader, leave=False):
        yhat = model(X)
        loss = torch.nn.CrossEntropyLoss()(yhat, y)
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            if sum(torch.sum(torch.isnan(p.grad))
                   for p in model.parameters()) == 0:
                opt.step()
        total_err += (yhat.max(dim=1).indices != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        if monitor is not None:
            total_monitor += monitor(model)
    return total_err / len(loader.dataset), total_loss / len(
        loader.dataset), total_monitor / len(loader)


for i in range(10):
    if i % 5 == 0:
        optim.param_groups[0]['lr'] = 1e-2
    epoch_err, epoch_loss, epoch_iters = epoch(
        train_dataloader, model, opt=optim, monitor=lambda m: m[2].iterations)
    print(
        f'Train loss = {epoch_loss} err = {epoch_err} iterations = {epoch_iters}'
    )
    epoch_err, epoch_loss, epoch_iters = epoch(
        test_dataloader, model, opt=None, monitor=lambda m: m[2].iterations)
    print(
        f'Test loss = {epoch_loss} err = {epoch_err} iterations = {epoch_iters}'
    )
