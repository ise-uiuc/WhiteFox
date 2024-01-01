
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 10, 5, stride=1)
        self.conv_2 = torch.nn.Conv2d(10, 1, 5, stride=1)
    def forward(self, x):
        v1 = self.conv_1(x)
        v2 = self.conv_2(v1)
        v3 = v2 > 0
        v4 = v2 * -0.5577
        v5 = torch.where(v3, v2, v4)
        return v5
# Inputs to the model
x2 = torch.randn(1, 1, 2, 2, device='cuda')
