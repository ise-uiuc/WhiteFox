
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.randn_like(torch.zeros_like(x1))
        return x2
# Inputs to the model
x1 = torch.randn(1)
