
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        a = torch.mean(torch.mul(x1, x2)).mul_(2).sum(0).squeeze_(-1)
        a += a
        return a
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
