
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1, groups=2)
        self.conv_1 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1, groups=4)
        self.conv_2 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1, groups=8)
    def forward(self, x1):
        v1 = self.conv_0(x1)
        v1 = torch.flatten(v1, 1)
        v1 = v1 - 0.5
        v1 = torch.reshape(v1, (-1, 8, 2, 18, 18))
        v1 = v1.permute(0, 2, 1, 3, 4)
        v1 = v1.reshape(v1.shape[0] * v1.shape[1], v1.shape[2], 18, 18)
        v1 = self.conv_1(v1)
        v1 = torch.flatten(v1, 1)
        v1 = v1 - 0.5
        v1 = torch.reshape(v1, (-1, 4, 4, 9, 9))
        v1 = self.conv_2(v1)
        v1 = torch.flatten(v1, 1)
        v1 = v1 - 0.5
        v1 = torch.reshape(v1, (-1, 2, 4, 9, 9))
        v1 = self.conv_0(v1)
        v1 = torch.flatten(v1, 1)
        v1 = v1 - 0.5
        v1 = torch.reshape(v1, (-1, 8, 2, 18, 18))
        v1 = self.conv_1(v1)
        v1 = torch.flatten(v1, 1)
        v1 = v1 - 0.5
        v1 = torch.reshape(v1, (-1, 4, 4, 9, 9))
        v1 = self.conv_2(v1)
        v1 = torch.flatten(v1, 1)
        v1 = v1 - 0.5
        v1 = torch.reshape(v1, (-1, 2, 4, 9, 9))
        return v1
# Inputs to the model
x1 = torch.randn(2, 3, 128, 128)
