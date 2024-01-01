
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1_1, x1_2, x3):
        v1 = x1_1
        v2 = torch.conv2d(v1, self.conv.weight.transpose(0, 1), None, self.conv.stride, (self.conv.padding[0] // 2, self.conv.padding[1] // 2), self.conv.dilation, self.conv.groups)
        v3 = torch.relu(v2)
        v4 = torch.conv2d(v3, self.conv.weight.transpose(0, 1), None, self.conv.stride, (self.conv.padding[0] // 2, self.conv.padding[1] // 2), self.conv.dilation, self.conv.groups)
        v5 = torch.relu(v4)
        v6 = self.conv(x1_2)
        v7 = v3 + x3
        v8 = torch.relu(v7)
        v9 = v5 + v6
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x1_1 = torch.randn(1, 16, 64, 64)
x1_2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
