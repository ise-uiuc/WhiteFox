
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=(1, 43), stride=1, padding=(1, 22))
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 1
        return v2
# Inputs to the model
x = torch.randn(8, 1, 34, 44)
