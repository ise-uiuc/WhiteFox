
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.randn(x1)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
