
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, padding1=None):
        v1 = torch.nn.functional.interpolate(x1, x2.shape[2:], mode='bicubic')
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        v2 = v1 + x2
        return v2
# Inputs to the model
x1 = torch.randn((1, 3, 32, 32))
x2 = torch.randn((1, 64, 32, 32))
