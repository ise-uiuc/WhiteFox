
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.dropout1 = torch.nn.Dropout2d(0.25)
        self.relu1 = torch.nn.ReLU()
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = v1 + v2
        v4 = self.conv3(x) + self.conv4(x)
        v5 = self.conv5(x) + v3
        v6 = v5 + v4
        v7 = self.dropout1(v6)
        v8 = self.relu1(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
