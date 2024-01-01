
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 32)

    def forward(self, x1, x2):
        v1 = self.fc1(x1 + x2)
        v2 = torch.t(torch.relu(v1))
        return v2

# Initializing the model
m = Model()
# Inputs to the model
x1 = torch.randn(32, 16)
x2 = torch.randn(16, 32)
