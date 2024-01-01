
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, other=None, other_padding=None, other_pad=None):
        v1 = self.conv(x1)
        if other == None:
            other_shape = [x1.shape[0]]
            for item in v1.shape:
                other_shape.append(item)
            other = torch.randn(other_shape)
            if other_padding == None:
                other_padding = [0, 0]
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 608, 608)
