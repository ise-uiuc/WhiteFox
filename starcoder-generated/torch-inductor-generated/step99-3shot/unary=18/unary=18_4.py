
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(1, 1), stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 16, kernel_size=(5, 5), stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(16, 4, kernel_size=(3, 3), stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(4, 1, kernel_size=(4, 4), stride=1, padding=0)
        self.maxpool1d1 = torch.nn.MaxPool1d(kernel_size=79, stride=79, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv4(v6)
        return self.maxpool1d1(v7)
# Inputs to the model
x1 = torch.randn(1, 1, 412)
