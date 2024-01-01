
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module0 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 6, 2, stride=2, padding=0)
 
    def forward(self, x3):
        v3 = self.conv2(self.module0(x3))
        v2 = v3
        v4 = torch.relu(v2)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(1, 3, 512, 512)
