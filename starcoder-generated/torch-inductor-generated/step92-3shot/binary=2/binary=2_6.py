
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 5, kernel_size=(1, 1))
    def forward(self, x):
        v1 = self.conv(x)
        const = torch.tensor(0.19186674)
        v2 = v1 - const
        return v2
# Inputs to the model
x = torch.randn(1, 1, 26, 26)
