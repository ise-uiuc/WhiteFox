
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.dropout2d = torch.nn.Dropout2d()
        self.linear = torch.nn.Linear(320*18*18, 64*9*9)
        self.dropout1d = torch.nn.Dropout(0.3)
        self.conv1 = torch.nn.Conv2d(32, 64, 5, stride=1, padding=2)

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.dropout2d(v2)
        v4 = self.linear(v3)
        v5 = self.dropout1d(v4)

        v6 = v5.reshape(v5.size()[0], 32, 9, 9) # reshape the linear output into 9x9 map

        v7 = self.conv1(v6)
        v8 = torch.relu(v7)

        v9 = torch.flatten(v8, start_dim=1)
        v10 = self.linear(v9)

        return v10
# Inputs to the model
x1 = torch.randn(16, 1, 128, 128)
