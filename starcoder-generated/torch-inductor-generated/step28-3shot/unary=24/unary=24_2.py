
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 64, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 20, 1, stride=1, padding=0)
        self.dropout1 = torch.nn.Dropout(p=0.2)
        self.dropout2 = torch.nn.Dropout(p=0.1)
        self.dropout3 = torch.nn.Dropout(p=0.3)
        self.linear1 = torch.nn.Linear(20, 64)
    def forward(self, x):
        negative_slope = 0.45
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 > 0
        v5 = v3 * negative_slope
        v6 = torch.where(v4, v3, v5)
        v7 = self.dropout1(v6)
        v8 = self.linear1(v7)
        v9 = self.dropout2(v8)
        v10 = self.dropout3(v9)
        v11 = self.linear1(v10)
        return v11
# Inputs to the model
x1 = torch.randn(2, 1, 128, 128)
