
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=(15, 15))
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 0.009614633237345492
        return v2
# Inputs to the model
x = torch.randn(1, 1, 70, 70)
