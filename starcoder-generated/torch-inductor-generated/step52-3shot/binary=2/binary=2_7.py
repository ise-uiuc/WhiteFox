
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=(1, 1), padding=(2, 1), stride=(4, 1))
    def forward(self, x3):
        v1 = self.conv(x3)
        v2 = v1 - "foo"
        return v2
# Inputs to the model
x3 = torch.randn(1, 3, 100, 100)
