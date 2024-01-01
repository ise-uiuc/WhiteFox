
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x):
        # TODO(you): replace torch.randn with input tensor x
        v1 = self.conv(x) # torch.randn(1, 2, 64, 64)
        v2 = v1 - torch.randn(3) # torch.randn(1, 2, 64, 64)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
