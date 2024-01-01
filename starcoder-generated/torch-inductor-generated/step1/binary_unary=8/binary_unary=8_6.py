
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        return F.relu(self.conv(x), other=torch.tensor(0.5))

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
