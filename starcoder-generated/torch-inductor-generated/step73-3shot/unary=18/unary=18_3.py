
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(11, (64 * 32) // 16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        v7 = torch.nn.Linear(64, 10)
        v8 = torch.sigmoid(v7)
        return v8
# Inputs to the model
x2 = torch.randn(1, 11, 32, 32)
