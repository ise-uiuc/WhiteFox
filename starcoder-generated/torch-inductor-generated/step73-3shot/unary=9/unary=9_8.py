
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x1):
        x2 = 3 + x1
        x3 = torch.clamp(x2, min=0)
        x4 = torch.clamp(x3, max=6)
        return x4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
