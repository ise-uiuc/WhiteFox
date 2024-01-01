
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=7, stride=2, padding=3),
            torch.nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            torch.nn.Conv2d(16, 4, kernel_size=3, stride=2, padding=1),
            torch.nn.Conv2d(4, 32, kernel_size=7, stride=1, padding=3),
            torch.nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
        )
        self.avgpool1 = torch.nn.AdaptiveAvgPool2d(1)
        self.conv2 = torch.nn.Conv2d(32, 16, kernel_size=1)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(16 * 1 * 1, 128)
        self.dropout = torch.nn.Dropout()
        self.linear2 = torch.nn.Linear(128, 10)
    def forward(self, x1):
        v1 = self.block1(x1)
        v2 = self.avgpool1(v1)
        v3 = self.conv2(v1)
        v4 = self.flatten(v2)
        v5 = self.linear1(v4)
        v6 = self.dropout(v5)
        v7 = self.linear2(v6)
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(128, 1, 224, 224)
