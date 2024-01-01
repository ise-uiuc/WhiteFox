
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = y2 - self.conv(x1) # Y2 is another input tensor passed by the user
        v2 = v1 - 0.25
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
