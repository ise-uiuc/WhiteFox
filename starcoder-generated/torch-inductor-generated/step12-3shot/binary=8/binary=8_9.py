
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, other):
        v1 = self.conv(x1)
        v2 = other + v1
        return v2

# Initializing the model
m = Model()

# Inputs to the model, other must be an InputParameter object in the signature
x1 = torch.randn(1, 3, 64, 64)
other = torch.randn(1, 8, 64, 64)
