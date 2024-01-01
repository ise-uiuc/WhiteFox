
class Model1(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.abs(x)
        x = torch.nn.functional.interpolate(x, mode=nn.Upsample.NEAREST)
        return x
# Inputs to the model
x = torch.randn(1, 1, 2, 2)
