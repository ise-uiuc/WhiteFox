
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 1, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        return v1.relu(other=0.25)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8, 122, 122)
