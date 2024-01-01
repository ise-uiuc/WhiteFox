
class Model(torch.nn.Module):
    def __init__(self, fallback_random=False):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 2)
        self.batchnorm = torch.nn.BatchNorm2d(2)
    def forward(self, x):
        x1 = self.batchnorm(self.conv(x))
        x2 = x1 + self.conv(x)
        x3 = torch.rand_like(x2) if not fallback_random else None
        x4 = x3 if x3 is not None else self.conv(x1)
        x5 = torch.nn.functional.dropout(x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
# Model begins

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x0):
        x = self.linear1(x0)
        x = self.relu(x)
        x = self.linear2(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 2)
