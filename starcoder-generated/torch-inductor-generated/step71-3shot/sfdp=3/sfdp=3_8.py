
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.ConvTranspose2d(3, 24, 4, stride=2, padding=1, bias=True)
        self.conv2 = torch.nn.ConvTranspose2d(24, 3, 4, stride=2, padding=1, bias=True)
 
    def forward(self, x1, x2):
        v1 = self.relu(self.conv1(x1))
        v2 = self.conv2(v1*0.5)
        v3 = v2 + x2
        v4 = v3 - 2
        v5 = v4 * x1
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
