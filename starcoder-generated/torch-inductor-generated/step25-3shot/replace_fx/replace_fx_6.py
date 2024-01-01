
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        z1 = torch.nn.functional.glu(x, dim=-1) + torch.nn.functional.glu(y, dim=-2)
        w1 = torch.nn.functional.gelu(z1)
        p1 = torch.nn.functional.glu(x) + torch.nn.functional.glu(w1)
        return torch.nn.functional.gelu(p1)
# Inputs to the model
x = torch.randn(10, 20)
y = torch.randn(10, 20)
