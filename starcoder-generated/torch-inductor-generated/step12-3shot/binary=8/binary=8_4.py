
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        
    def forward(self, x1, w=None):
        if w is not None:
            output = self.conv(x1) + w
        else:
            output = self.conv(x1)
        return output
        
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
