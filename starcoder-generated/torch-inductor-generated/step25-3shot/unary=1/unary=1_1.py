
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 4)
        self.fc2 = torch.nn.Linear(4, 2)
        torch.nn.init.uniform_(self.fc1.weight, -0.005, 0.005)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.uniform_(self.fc2.weight, -0.005, 0.005)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = x + (x * x * x) * 0.044715
        x = torch.tanh(x)
        v1 = self.fc1(x)
        v2 = torch.tanh(v1)
        v3 = v2 + 1
        v4 = self.fc2(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
