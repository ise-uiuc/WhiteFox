
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.nn.functional.sigmoid(x1)
        x3 = x2 + 1
        return x3
# Inputs to the model
x1 = torch.randn(2, 2, 2)
