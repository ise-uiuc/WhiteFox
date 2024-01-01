
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1)
        self.conv2 = torch.nn.Conv2d(3, 3, 1)
    def forward(self, x1, other=None):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        if other == None:
            other1_shape = [1]
            for item in v1.shape:
                other1_shape.append(item)
            other1 = torch.randn(other1_shape)
            other2_shape = [1]
            for item in v2.shape:
                other2_shape.append(item)
            other2 = torch.randn(other2_shape)
        v3 = v1 + other1
        v4 = v2 + other2
        v5 = torch.abs(v3 + v4)

        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
