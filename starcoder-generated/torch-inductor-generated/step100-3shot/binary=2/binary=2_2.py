
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 16, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 0.7854
        return v2
# Inputs to the model
x = torch.randn(1, 8, 32, 32)
