
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.sum(x1, dim=2)
        v2 = torch.sqrt(v1)
        return v2 + x1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
