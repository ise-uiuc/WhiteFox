
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand(1, dtype=torch.double, device=x1.device)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.rand_like(x1)
