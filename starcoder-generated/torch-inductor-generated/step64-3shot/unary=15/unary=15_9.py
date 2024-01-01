
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 5), stride=(2, 3), padding=(2, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=(3, 5), stride=(1, 3), padding=(1, 2))
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv5 = torch.nn.Conv2d(128, 256, 1, 1, 0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = torch.relu(v4)
        v6 = self.conv5(v5)
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
