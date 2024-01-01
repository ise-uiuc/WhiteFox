
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1= torch.nn.Conv2d(3, 11, 1, stride=1, padding=0)
        self.conv2= torch.nn.Conv2d(11, 9, 6, stride=1, padding=1)
        self.conv3= torch.nn.Conv2d(9, 6, 1, stride=1, padding=0)
        self.conv4= torch.nn.Conv2d(6, 7, 1, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1 = torch.nn.functional.interpolate(v1, scale_factor=2.0)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v4 = torch.nn.functional.interpolate(v4, scale_factor=0.5)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v7 = torch.nn.functional.interpolate(v7, scale_factor=0.5)
        v8 = self.conv3(v7)
        v8 = torch.nn.functional.interpolate(v8, scale_factor=0.5)
        v9 = self.conv4(v8)
        v9 = torch.nn.functional.interpolate(v9, scale_factor=0.5)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
