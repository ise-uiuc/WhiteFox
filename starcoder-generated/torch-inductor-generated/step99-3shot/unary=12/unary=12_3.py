
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.relu(v1)
        v3 = F.max_pool2d(v2, padding=[0, 2, 2, 0], stride=[1, 1, 1, 1], kernel_size=[2, 2], dilation=[1, 1])
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
