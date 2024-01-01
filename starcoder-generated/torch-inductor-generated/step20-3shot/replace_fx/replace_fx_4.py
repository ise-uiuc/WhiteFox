
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x3 = torch.randn_like(x2)
        return x3
# Inputs to the model
x1 = torch.randn(2)
x2 = torch.randn(1, 2)
