
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)
        self.fc = torch.nn.Linear(4, 16)
 
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 =  self.fc(x2)

        v3 = v1 + v2

        return v3

# Initializing the model
m = Model()
m.eval()

# Inputs to the model
x1 = torch.randn(1, 3, 480, 640)
x2 = torch.randn(1, 4, 160, 1920)
