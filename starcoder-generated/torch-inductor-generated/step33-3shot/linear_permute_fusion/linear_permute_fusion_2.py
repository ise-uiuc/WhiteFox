
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.relu(x1)
        v3 = self.linear(v1)
        v4 = v1.permute(0, 2, 1)
        v5 = v4 - v3
        v6 = torch.tanh(v3)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
