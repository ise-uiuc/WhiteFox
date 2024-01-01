
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        return v1 + other, v1, other

# Initializing the model
x1 = torch.randn(1, 3, 64, 64)
other = torch.randint(low=0, high=256, size=(1, 1, 1, 1))
model = Model()

# Inputs to the model
__output1__, __output2__, __output3__ = model(x1, other=other)

