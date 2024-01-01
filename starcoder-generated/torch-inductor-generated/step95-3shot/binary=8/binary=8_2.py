
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, padding=1, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 4, 3, padding=1, stride=1)
        self.conv3 = torch.nn.Conv2d(3, 4, 3, padding=1, stride=1)
        self.conv4 = torch.nn.Conv2d(3, 4, 3, padding=1, stride=1)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v_cat = torch.cat([v1, v2], dim=1)
        v3 = self.conv3(x3)
        v4 = self.conv4(x4)
        v_cat = torch.cat([v_cat, v3, v4], dim=1)
        v_out = torch.cat([v_cat, v_cat], dim=1)
        return v_out
# Inputs to the model
x1 = torch.randn(1, 3, 30, 34)
x2 = torch.randn(1, 3, 30, 34)
x3 = torch.randn(1, 3, 30, 34)
x4 = torch.randn(1, 3, 30, 34)
