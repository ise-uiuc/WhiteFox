
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(256, 512, kernel_size, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(512, 1, kernel_size, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 256, 14, 14)


