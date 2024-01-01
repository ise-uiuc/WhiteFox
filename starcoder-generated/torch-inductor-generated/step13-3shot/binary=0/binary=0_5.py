
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(48, 64, 3, stride=1, padding=2)
    def forward(self, x1, other=None, some_parameter=1):
        v1 = self.conv(x1)
        if some_parameter < 2:
            if other == None:
                other = torch.randn(v1.shape)
        elif other == None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 48, 64, 64)
