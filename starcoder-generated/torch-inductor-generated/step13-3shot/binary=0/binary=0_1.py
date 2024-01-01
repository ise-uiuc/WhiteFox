
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 5, 1, stride=1, padding=0)
    def forward(self, x1, some_parameter=True, other=None):
        v1 = self.conv(x1)
        if some_parameter == False:
            some_parameter = torch.randn(v1.shape)
            return v1 + some_parameter
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 1, 1)
