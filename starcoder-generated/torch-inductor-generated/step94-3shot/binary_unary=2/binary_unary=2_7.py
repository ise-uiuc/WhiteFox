
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(3, 64, kernel_size=(3))
        self.conv2d = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv = torch.nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.batch_norm2d = torch.nn.BatchNorm2d(64)

    def forward(self, x1):
        v1 = self.conv1d(x1)
        v2 = F.gelu(v1)
        v3 = self.conv2d(v2)
        v4 = F.gelu(v3)
        v5 = self.conv(v4)
        v6 = F.gelu(v5)
        v7 = self.bn1(v6)
        v8 = self.batch_norm2d(v7)
        v9 = v8 - 1
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 256)
