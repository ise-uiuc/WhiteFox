
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a):
        b = a
        c = b**2
        return (c, a)
# Inputs to the model
x1 = torch.randn(32)
