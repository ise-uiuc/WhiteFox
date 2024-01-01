
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(5, 5)
        self.bn = torch.nn.BatchNorm2d(5)
        self.relu = torch.nn.ReLU6()
        self.conv = torch.nn.Conv2d(5, 5, 1, stride=1, padding=0)
 
    def forward(self, x):
        v1 = self.fc(x)
        v2 = self.bn(v1)
        v3 = self.relu(v2)
        v4 = self.conv(v3).contiguous()
        v5 = v1 * v4
        v6 = v5.contiguous()
        v7 = v6 / 6
        v8 = v7.float()
        return v8
# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 5).contiguous()
