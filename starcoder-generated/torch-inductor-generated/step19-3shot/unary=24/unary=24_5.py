
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 16, kernel_size=(5, ), stride=(2, ))
    def forward(self, x):
        v1 = self.conv(x)
        mask = v1 > 0
        v2 = v1 * 0.1
        v3 = torch.where(mask, v1, v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 5, 40, 80)
