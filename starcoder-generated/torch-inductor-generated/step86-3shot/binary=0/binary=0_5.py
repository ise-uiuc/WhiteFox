
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 32, 1, stride=1, padding=1)
    def forward(self, x, padding_mode='zeros'):
        v1 = self.conv(x)
        if padding_mode == None:
            padding_mode = 'zeros'
        v2 = v1 + padding_mode
        return v2
# Inputs to the model
x = torch.randn(1, 16, 16, 16)
