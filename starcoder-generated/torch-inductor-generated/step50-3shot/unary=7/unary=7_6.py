
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.fc = torch.nn.Linear(8 * 64 * 64, 10)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        b, c, h, w = v1.shape
        v1 = torch.flatten(v1, 1, -1).transpose(1, 2)
        v2 = self.fc(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
