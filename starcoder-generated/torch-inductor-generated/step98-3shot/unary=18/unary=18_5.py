
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(256, 64, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0))
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.conv3 = torch.nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 256, 1))
