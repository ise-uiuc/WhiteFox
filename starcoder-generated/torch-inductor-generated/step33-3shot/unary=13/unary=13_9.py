
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 8, bias=False)
        self.conv = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1, bias=False)
 
    def forward(self, x1):
        w = 1/(1 + 0.25*torch.max(torch.abs(self.fc1.weight)))
        v = self.fc1(x1)
        v = v * w
        v = self.conv(v)
        v = v * w

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
