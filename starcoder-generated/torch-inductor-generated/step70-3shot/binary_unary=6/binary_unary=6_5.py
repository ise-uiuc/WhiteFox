
class Model(torch.nn.Module):
    def __init__(self, channels, other):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, 8, 1, stride=1, padding=1)
        self.fc = torch.nn.Linear(8, 32)
        self.other = other
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = v1.reshape((v1.size()[0], -1))
        v1 = self.fc(v1)
        v2 = v1 - self.other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model(3, 3.1415)
 
# Inputs to the model
x1 = torch.randn(1, 3, 8, 8)
