
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
    def forward(self, x1, padding1='default'):
        v1 = self.conv(x1)
        if padding1 == 'default':
            padding1 = torch.randn(v1.shape)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
