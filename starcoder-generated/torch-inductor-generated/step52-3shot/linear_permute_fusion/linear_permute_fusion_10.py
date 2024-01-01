
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v0 = self.linear(x1)
        v1 = v0.permute(0, 2, 1)
        v2 = torch.flatten(v1, 2)
        return v0 + v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
