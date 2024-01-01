
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.fc = torch.nn.Linear(512, 10)
 
    def forward(self, x1):
        v0 = self.conv(x1)
        v1 = v0.reshape(v0.shape[0], -1)
        v2 = self.fc(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
