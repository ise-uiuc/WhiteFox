
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(16, 3, 1, stride=1, padding=1)
    def forward(self, x):
        # Create an 'add' module and apply it on 'x'
        res = x + 32.5
        v1 = self.conv2(res)
        x = torch.neg(x)
        v2 = self.conv2(x)
        v3 = v2 - 4.3
        return v3
# Inputs to the model
x = torch.randn(1, 16, 7, 10)
