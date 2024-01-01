
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=(4,1), stride=(3,2), padding=(1,2))
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 36.4
        return v2
# Inputs to the model
x = torch.randn(1, 3, 32, 64)
