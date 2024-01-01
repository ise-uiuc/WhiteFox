
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 5, stride=1, dilation=3, padding=2)
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        if other == None:
            other_shape = [v1.shape[0]]
            for item in v1.shape:
                other_shape.append(item)
            other = torch.randn(other_shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(3, 3, 255, 255)
