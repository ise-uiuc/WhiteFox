
class Module1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 32)
    def forward(self, x):
        return self.fc1(x)
class Module2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(32, 64)
    def forward(self, x):
        return self.fc1(x)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 64)
        self.module1 = Module1()
        self.module2 = Module2()
    def forward(self, x):
        x = self.fc1(x)
        x = self.module1(x)
        x = self.module2(x)
        return x
# Inputs to the model
x = torch.randn(2, 16)
