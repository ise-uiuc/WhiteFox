
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, stride=1, padding=3)
        self.fc = torch.nn.Linear(2048, 2304)
        self.dropout1 = torch.nn.Dropout(p=0.3)
        self.dropout2 = torch.nn.Dropout(p=0.2)
        self.dropout3 = torch.nn.Dropout(p=0.1)
    def forward(self, x1):
        v0 = self.conv1(x1)
        v0 = torch.sigmoid(v0)
        v1 = self.conv2(v0)
        v1 = torch.sigmoid(v1)
        v2 = self.conv3(v1)
        v2 = torch.cos(v2)
        v3 = torch.matmul(v2, v1.permute([0,2,1,3]))
        v3 = nn.ReLU()(v3)
        v4 = self.fc(v3)
        v4 = torch.sigmoid(v4)
        v5 = self.dropout1(v4)
        v5 = self.dropout2(v5)
        v6 = self.dropout3(v5)
        return nn.ReLU()(v6)
# Inputs to the model
x1 = torch.randn(1, 8, 28, 28)
