
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 9, stride=1, padding=4)
        self.conv2 = torch.nn.Conv2d(16, 32, 5, stride=1, padding=2)
        self.fc1 = torch.nn.Linear(8192, 32)
    def forward(self, x1):
        v1 = self.conv1(x1)
        t2 = self.conv2(v1)
        v3 = t2.view(-1, 8192)
        v4 = self.fc1(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
