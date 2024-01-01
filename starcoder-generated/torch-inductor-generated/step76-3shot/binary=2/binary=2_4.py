
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
    def forward(self, x4):
        v1 = self.conv(x4)
        v2 = v1 - True
        return v2
# Inputs to the model
x4 = torch.randn(1, 3, 25, 25)
