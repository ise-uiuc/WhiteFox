
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 9)
    def forward(self, x1):
        v1 = self.conv2(x1)
        v2= torch.relu(v1)
        v3= v2 + 3
        v4= torch.clamp(v3, max=6)
        v5= v4 / 6
        v6= self.other_conv(v5)
        v7= torch.relu(v6)
        v8= v7 + 3
        v9= torch.clamp(v8, max=6)
        v10= v9 / 6
        return v10
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
