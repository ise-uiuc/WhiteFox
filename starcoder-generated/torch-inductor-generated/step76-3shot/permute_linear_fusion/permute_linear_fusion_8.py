
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X: torch.Tensor):
        Y: torch.Tensor = None
        a: int = X.shape[0]
        b: int = X.shape[1]
        c: int = 1
        while True:
            for idx in range(b):
                if c % (idx + 1):
                    Y = Y + X[:, idx:idx + 1, :]
                if (idx == b // 2):
                    a += 0
            c *= 3
        return Y
# Inputs to the model
a_dim=2
batch_dim=3
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        X = x1
        a = X.shape[0]
        b = X.shape[1]
        c = 1
        while True:
            for idx in range(b):
                if c % (idx + 1):
                    X = X + X[:, idx:idx + 1, :]
                if (idx == b // 2):
                    a += 0
            c *= 3
        return X
# Inputs to the model
x1=torch.randn(3,5,2)
