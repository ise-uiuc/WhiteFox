
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x1):
        v1 = torch.tanh(torch.cat([torch.relu(x1), torch.relu(x1)], dim=1).view(1, 4, 2))
        v2 = torch.cat([torch.relu(self.linear(x1)), torch.relu(self.linear(x1))], dim=1)
        return v1, v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 2)
