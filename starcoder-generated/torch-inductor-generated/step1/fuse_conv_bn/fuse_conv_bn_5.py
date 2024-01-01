
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, 3, stride=2, padding=3, dilation=1, groups=1, bias=False, padding_mode='zeros')

    def forward(self, x1):
        x2 = F.conv2d(x1, self.conv.weight)
# Inputs to the model
        return x2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 5, 5)
