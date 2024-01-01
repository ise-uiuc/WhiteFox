
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(512, 1000, 1, stride=1, padding=0)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(1000, 1000)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.avg_pool(v1)
        v3 = v2.flatten(start_dim=1)
        v4 = self.dropout(v3)
        v5 = self.fc(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 512, 28, 28)
