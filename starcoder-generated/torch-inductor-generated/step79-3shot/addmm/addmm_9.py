
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, a):
        a = a.detach().requires_grad_(True)
        return x * a
# Inputs to the model
x1 = torch.randn(3, 3)
a = torch.randn(3, 3)
