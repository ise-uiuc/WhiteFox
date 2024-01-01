
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        x3 = torch.rand_like(x1) # No pattern
        x4 = torch.rand_like(x1, device='cpu') # No pattern
        return x2 + x3 + x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
