
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        z1 = x1 * x1
        z2 = torch.rand_like(z1)
        y1 = torch.nn.functional.gelu(z1)
        y2 = torch.nn.functional.gelu(z2)
        y3 = torch.nn.functional.gelu(z1)
        w1 = y1 + y2 + y3
        v1 = w1 + torch.nn.functional.gelu(z1)
        w2 = torch.rand_like(x1)
        w3 = torch.rand_like(w1)
        y4 = w1 + w2 + w3
        y5 = torch.nn.functional.gelu(w1)
        return y4
# Inputs to the model
x1 = torch.randn(4, 64, 64)
