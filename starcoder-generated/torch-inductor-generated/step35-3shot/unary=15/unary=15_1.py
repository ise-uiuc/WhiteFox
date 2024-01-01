
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.dropout2 = torch.nn.Dropout(p=0.5)
        self.conv3 = torch.nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.dropout3 = torch.nn.Dropout(p=0.5)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.dropout(v1)
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = self.dropout2(v4)
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = self.dropout3(v7)
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(4, 3, 200, 300)
