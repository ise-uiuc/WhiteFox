
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x2):
        v2 = x2
        v0 = self.linear(v2)
        y = self.relu(v0)
        q0 = torch.relu(self.linear.weight)
        v1 = y.permute(0, 2, 1)
        return v1
# Inputs to the model
x2 = torch.randn(1, 2, 2)
