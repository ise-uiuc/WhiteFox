
class Model(torch.nn.Module):
    def __init__(self, conv_op):
        super().__init__()
        self.conv = conv_op
 
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = v1 + x2
        return v2

# Initializing the model
conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
m = Model(conv_op=conv)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 8, 64, 64)
