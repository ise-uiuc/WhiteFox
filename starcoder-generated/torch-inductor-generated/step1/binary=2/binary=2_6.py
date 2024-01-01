
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x, y):
        v1 = self.conv(x)
        v2 = v1 - y
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)  # input_of_conv
y = torch.randn(1, 8, 62, 62)  # other
