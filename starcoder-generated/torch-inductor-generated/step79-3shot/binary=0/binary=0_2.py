
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 11, 5, stride=1, padding=1)
    def forward(self, x1, bias1=None, other=None, bias3=None, some_parameter=1, some_parameter_2=None, bias_4=False, padding3=False, bias_5=None, padding=None):
        v1 = self.conv(x1)
        if bias1 == True or other == True:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 2, 3)
