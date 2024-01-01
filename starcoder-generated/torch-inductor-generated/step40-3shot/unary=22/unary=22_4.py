
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        v1 = self.conv3(x1)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
