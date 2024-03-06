
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = v1 + x2
        return v2

# Initializing the model
m = Model()

# Initializing the tensors with the same input tensor
x1 = torch.randn(1, 3, 64, 64)
x2 = x1

# Inputs to the model