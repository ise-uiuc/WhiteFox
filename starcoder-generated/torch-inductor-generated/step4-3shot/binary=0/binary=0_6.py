
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        if other == None:
            other_shape = [1]
            for item in v1.shape[1:]:
                other_shape.append(item)
            other = torch.randn(v1.shape[0], *other_shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
