
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = self.linear(x1)
        x2 = x1.detach()
        x3 = torch.neg(x2)
        v2 = x3.to(v1.dtype)
        v3 = v1 + v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 2)
