
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(57, 64, 3, stride=(2, 2), padding=(2, 2))
        self.conv2 = torch.nn.Conv2d(64, 32, 1, padding=0)
        self.conv2_1 = torch.nn.Conv2d(64, 16, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 1, 3, padding=1)
        self.fc = torch.nn.Linear(192, 1)
    def forward(self, x):
        v1 = self.conv1(x)
        v3_1 = self.conv2(v1)
        v5_1 = self.conv2_1(v1)
        v3 = v3_1 + v5_1
        v5 = torch.tanh(v3)
        v8 = self.conv3(v5)
        v11 = self.fc(v8)
        return v11
# Inputs to the model
x = torch.randn(2, 57, 560, 560)
