
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 - 0.4
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
