
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_linear = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, 1, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(6, 9, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.linear = torch.nn.Linear(180, 10)
 
    def forward(self, x1):
        v1 = self.pre_linear(x1)
        v3 = torch.flatten(v1, 1)
        v4 = self.linear(v3)
        return v4

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 14)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return v1

# Initializing the model
m = Model()
m1 = Model1()

# Inputs to the model
x1 = torch.randn(1, 3, 96, 96)
