
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1, x2):
        v1 = torch.nn.functional.relu(x1)
        v2 = x1.permute(0, 2, 1)
        v3, v4 = 0.7 * self.linear(v1), self.linear(v2)
        return v3, v4
# Inputs to the model
x1 = torch.randn(1, 3, 2)
x2 = torch.randn(1, 3, 2)
