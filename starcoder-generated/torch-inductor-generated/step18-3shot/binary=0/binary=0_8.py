
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 3, kernel_size=(1, 3), stride=(2, 1), padding=(0, 1))
    def forward(self, x1, other=1):
        v1 = self.conv(x1)
        v3 = v1 + other
        return v3
# Inputs to the model
x1 = torch.randn(2, 4, 64, 64)
