
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 16)
        self.relu = torch.nn.LeakyReLU(0.2)
        self.pool = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(4, 2, 1, stride=1, padding=1)
        self.dropout = torch.nn.Dropout(0.3)
    def forward(self, x1):
        v1 = self.dropout(x1)
        v2 = self.pool(v1)
        v3 = self.relu(self.fc(v2))
        v4 = v3.reshape(1, 4, 28, 28)
        v5 = self.conv2(v4)
        v6 = v5 + x1
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 32, 32)
