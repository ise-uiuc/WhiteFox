
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        a = torch.nn.Conv2d(1, 1, kernel_size=1)
    def forward(self, x1):
        v1 = torch.nn.functional.interpolate(x1, size=[1, 64, 64], mode='nearest')
        v2 = torch.nn.functional.interpolate(v1, size=[1, 64, 64], mode='nearest')
        return v2.permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(1, 1, 8, 8)
# Model end
