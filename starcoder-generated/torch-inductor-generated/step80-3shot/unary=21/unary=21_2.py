
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(3, 256, 1, padding=(0, 1), stride=1)
        self.conv2 = torch.nn.Conv2d(256, 256, (1, 7), padding=(0, 3), stride=1)
        self.conv3 = torch.nn.Conv2d(256, 3, (1, 5), padding=(0, 2), stride=1)
    def forward(self, x1):
        v2 = self.relu(x1)
        v3 = self.conv1(v2)
        v4 = self.relu(v3)
        v5 = self.conv2(v4)
        v6 = self.relu(v5)
        v7 = self.conv3(v6)
        v8 = self.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(4, 3, 256, 241)
