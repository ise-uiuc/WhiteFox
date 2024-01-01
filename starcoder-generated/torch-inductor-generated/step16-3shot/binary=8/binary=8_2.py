
class Model(torch.nn.Module):
    __constants__ = ['other']
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 224, strides=[2,2])
        self.other = torch.zeros((224, 3, 3, 3), dtype=torch.float32)
 
    def forward(self, x2):
        v7 = self.conv(x2)
        return v7 + self.other

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
