
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(15, 8, 1, stride=1, padding=1)
    def forward(self, x1, other='other', alpha='alpha'):
        v1 = self.conv(x1)
        if other == 'other':
            other = torch.randn(v1.shape)
        if alpha == 'alpha':
            alpha = torch.randn(v1.shape)
        v2 = alpha * (v1 + other)
        return v2
# Inputs to the model
x1 = torch.randn(1, 15, 64, 64)
