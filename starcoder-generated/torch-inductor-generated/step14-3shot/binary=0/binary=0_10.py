
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 27, 1, stride=1, padding=1)
    def forward(self, x1, other=None, dim1=None):
        v1 = self.conv(x1)
        if dim1 == None:
            dim1 = []
            for item in v1.shape:
                dim1.append(item)
            dim1.insert(-1, 3)
        v2 = torch.randn(v1.shape[:-1])
        if other == None:
            other = v2.repeat(dim1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
