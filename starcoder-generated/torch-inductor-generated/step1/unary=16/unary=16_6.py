
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = torch.nn.Linear(16, 32, bias=False)
        self.fc1 = torch.nn.Linear(32, 32, bias=False)

    def forward(self, x, y):
        x1 = self.fc0(x)
        x2 = x1 + y
        x4 = self.fc1(x2)
        x3 = torch.nn.functional.relu(x4)
        return x3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16)
y = torch.randn(1, 32)
