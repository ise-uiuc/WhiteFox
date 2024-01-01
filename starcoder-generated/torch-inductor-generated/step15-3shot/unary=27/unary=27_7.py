
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 8, 3, stride=2, padding=2)

        self.bn1 = torch.nn.BatchNorm2d(32)
        self.relu1 = torch.nn.ReLU()

        self.max_pool3d = torch.nn.MaxPool3d(kernel_size=2, stride=2)
    def forward(self, x1):
        v1 = self.conv(x1)

        v7 = v1 + torch.randn_like(v1)
        v8 = torch.relu_(v7)

        return v8
# Inputs to the model
x1 = torch.randn(1, 32, 100, 100)
