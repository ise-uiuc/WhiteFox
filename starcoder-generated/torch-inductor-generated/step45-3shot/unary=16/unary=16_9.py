
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(128, 32)
        self.linear_2 = torch.nn.Linear(32, 64)

    def forward(self, x1):
        v1 = self.linear_1(x1)
        v2 = torch.relu(v1)
        v3 = self.linear_2(v2)
        v4 = torch.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
