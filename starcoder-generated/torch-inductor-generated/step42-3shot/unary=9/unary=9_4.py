
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv06 = torch.nn.Conv2d(1, 1, kernel_size=(6, 1), stride=(6, 1), padding=(4, 0))
        self.conv01 = torch.nn.Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
    def forward(self, x1):
        x1 = self.conv06(x1)
        x1 = x1 + 0.3
        x1 = self.conv01(x1)
        x1 = x1.add(2.5)
        x1 = self.conv06(x1)
        x1 = x1 + 0.3
        x1 = self.conv01(x1)
        x1 = x1.add(2.5)
        return x1.clamp(min=0.0, max=6.0) / 6.0
# Inputs to the model
x1 = torch.randn(1, 1, 64, 1)
