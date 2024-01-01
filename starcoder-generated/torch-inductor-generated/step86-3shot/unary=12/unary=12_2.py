
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, stride=3, padding=1)
    def forward(self, x):
        x1 = self.conv(x)
        x2 = torch.sigmoid(x1)
        x3 = x1 * x2
        return x3
# Inputs to the model
x = torch.rand(1,3,64,64) # dummy input
