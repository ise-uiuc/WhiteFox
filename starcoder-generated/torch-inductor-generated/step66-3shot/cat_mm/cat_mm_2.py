
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x1 = x1.mm(x2) + x2
        x2 = x2.mm(x1) + x1
        return x1, x2
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
