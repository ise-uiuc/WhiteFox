
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        z = self.linear.weight
        v1 = x1.permute(0, 2, 1)
        v2 = self.linear(v1)
        return torch.nn.ReLU(v2)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
