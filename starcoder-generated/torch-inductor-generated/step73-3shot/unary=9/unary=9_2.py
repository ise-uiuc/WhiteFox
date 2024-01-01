
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3)
        self.conv2 = torch.nn.Conv2d(4, 4, 3)
    def forward(self, x):
        x = 3. + self.conv1(x)
        x = x.clamp(min=0.)
        x = x.clamp(max=6.)
        x = self.conv2(x)
        out = 3. + x
        out = out.clamp(min=0.)
        out = out.clamp(max=6.)
        return out
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
