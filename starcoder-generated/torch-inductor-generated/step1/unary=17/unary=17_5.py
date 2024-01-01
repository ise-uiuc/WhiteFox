
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 8, 2, stride=2)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = F.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(8, 3, 4, 4)
