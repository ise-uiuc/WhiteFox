
class Model(torch.nn.Module):
    def __init__(self, min_value):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 10, 4)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(3, stride=1, padding=1)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(1440, 500)
        self.tanh = torch.nn.Tanh()
        self.min_value = min_value
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = self.relu(v2)
        v4 = self.maxpool(v3)
        v5 = self.conv(x2)
        v6 = torch.clamp_min(v5, self.min_value)
        v7 = self.relu(v6)
        v8 = self.maxpool(v7)
        v9 = self.flatten(v4)
        v10 = self.flatten(v8)
        v11 = torch.add(v9, v10)
        v12 = self.linear(v11)
        v13 = self.tanh(v12)
        return v13
min_value = -0.1
# Inputs to the model
x1 = torch.randn(1, 2, 398, 398)
x2 = torch.randn(1, 2, 398, 398)
