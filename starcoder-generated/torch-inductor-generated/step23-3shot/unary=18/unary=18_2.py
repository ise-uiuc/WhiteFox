
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 5, 1, stride=1, padding=1)
    def forward(self, x1):
#         v1 = self.conv(x1) # add this line of code into forward()
        v1 = nn.functional.sigmoid(self.conv(x1))
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
