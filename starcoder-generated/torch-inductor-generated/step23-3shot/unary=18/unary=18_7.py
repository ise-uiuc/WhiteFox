
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace)
        self.conv5 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU(inplace)
        self.conv = torch.nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x1):
        v0 = self.conv1(x1)
        v1 = self.relu1(v0)
        v2 = self.conv2(v1)
        v3 = self.relu2(v2)
        v4 = self.conv3(v3)
        v5 = self.relu3(v4)
        v6 = self.conv4(v5)
        v7 = self.relu4(v6)
        v8 = self.conv5(v7)
        v9 = self.relu5(v8)
        v10 = self.conv(v9)
        v11 = self.sigmoid(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
