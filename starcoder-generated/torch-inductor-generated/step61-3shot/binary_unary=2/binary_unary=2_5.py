
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
    def forward(self, x):
        v1 = self.conv2d(x) - 12
        v2 = F.relu(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
