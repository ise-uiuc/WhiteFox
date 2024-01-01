
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x0, x1, x2, mask):
        x = x0 + x1
        x = x + x2
        y = x + mask
        return y
# Inputs to the model
x0 = torch.randn(1, 64, 56, 56)
x1 = torch.randn(1, 64, 56, 56)
x2 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
