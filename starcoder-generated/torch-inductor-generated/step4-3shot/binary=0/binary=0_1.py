
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 1, stride=1, padding=1)
    def forward(self, x, other=None):
        v1 = self.conv(x)
        if other == None:
            other_shape = [1]
            for item in v1.shape:
                other_shape.append(item)
            other = torch.randn(other_shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
