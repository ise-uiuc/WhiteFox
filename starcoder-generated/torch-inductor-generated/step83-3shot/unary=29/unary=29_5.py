
class Model(torch.nn.Module):
    def __init__(self, min_value=1, max_value=0):
        super().__init__()
        self.conv2d_1 = torch.nn.Conv2d(3, 94, 1, stride=1, padding=0)
        self.conv2d_2 = torch.nn.Conv2d(3, 30, 1, stride=1, padding=0)
        self.conv2d_3 = torch.nn.Conv2d(3, 79, 1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1, x2, x3):
        v1 = self.conv2d_1(x1)
        v2 = self.conv2d_2(x2)
        v3 = self.conv2d_3(x3)
        v4 = torch.cat((v1, v2, v3), 1)
        v5 = torch.clamp_min(v4, self.min_value)
        v6 = torch.clamp_max(v5, self.max_value)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 255, 255)
x2 = torch.randn(1, 3, 255, 255)
x3 = torch.randn(1, 3, 255, 255)
