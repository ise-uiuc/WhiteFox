
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.fc = torch.nn.Linear(256, 32)
 
    def forward(self, x, other):
        v1 = self.conv(x)
        other1 = self.fc(other)
        v2 = v1 - other1
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
other = torch.randn(1, 256)
