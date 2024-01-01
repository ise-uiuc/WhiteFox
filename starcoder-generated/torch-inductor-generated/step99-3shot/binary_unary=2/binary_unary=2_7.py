
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(1, 10, 3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(in_features=980, out_features=1280)
        self.drop1 = torch.nn.Dropout(p=0.2)
        self.fc2 = torch.nn.Linear(in_features=1280, out_features=256)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 2.1
        v3 = F.relu(v2)
        v4 = v3.flatten(1)
        v5 = self.fc1(v4)
        v6 = self.drop1(v5)
        v7 = F.relu(v6)
        v8 = self.fc2(v7)
        v9 = v8 - 0.01
        v10 = F.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
