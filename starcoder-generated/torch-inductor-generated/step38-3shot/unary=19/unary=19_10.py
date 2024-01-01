
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.pool = torch.nn.AvgPool2d(8, 8, 8)

        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)

        self.relu = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(1024, 512)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear2 = torch.nn.Linear(512, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x1):
        x2 = self.conv1(x1)
        x4 = self.pool(x2)

        x5 = self.bn1(x4)
        x6 = self.relu(x5)

        x7 = self.bn2(x6)
        x8 = self.relu(x7)

        x8 = x8.view(x8.size(0), -1)
        x9 = self.linear1(x8)

        x10 = self.dropout(x9)
        x11 = self.linear2(x10)

        x12 = self.sigmoid(x11)

        return x12

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
