
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        conv2 = nn.Conv2d(32, 64, 3, 1)
        bn1 = nn.GroupNorm(32, 64) # bn1 has 32 groups
        bn2 = nn.BatchNorm2d(64)
        self.submodel = torch.nn.Sequential(conv2, bn2)
        self.linear = nn.Linear(9216, 128)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(3200, 3200)
        self.fc2 = nn.Linear(3200, 10)
    def forward(self, x, y):
        x1 = self.conv1(x)
        x1 = self.submodel(x1)
        x2 = self.activation(x1)
        x2 = torch.flatten(x2, 1)
        x2 = self.linear(x2)
        x2 = self.dropout(x2)
        z1 = self.fc1(x2)
        z2 = self.fc2(z1)
        z = z1 + z2
        return z
# Inputs to the model
x = torch.randn(4, 1, 28, 28)
y = torch.randn(4, 1, 28, 28)
