
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.nn.functional.dropout_(x)
        return x
# Inputs to the model
x1 = torch.randn(1)
x2 = torch.randn(1)
x3 = torch.randn(1)
