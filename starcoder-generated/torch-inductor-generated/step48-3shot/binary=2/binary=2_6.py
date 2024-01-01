
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(4, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm3d(32)
        self.conv2 = torch.nn.Conv3d(32, 32, kernel_size=1, stride=(1, 1, 1), padding=0, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x4):
        v1 = x4.to(torch.float16)
        v2 = self.conv1(v1)
        v3 = self.bn1(v2)
        v4 = v3.to(torch.float32)
        v5 = self.conv2(v4)
        v6 = self.bn2(v5)
        return self.sigmoid(v6)
# Inputs to the model
x4 = torch.randn(1, 4, 160, 5)
