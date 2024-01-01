
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        x1 = self.conv(x)
        v6 = torch.nn.functional.relu(x1)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
