
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(128, 32, 5, stride=2, padding=2)
    def forward(self, x1, padding1=None, other=3):
        if padding1 == None:
            padding1 = torch.randn(32, 32).numpy()
        v1 = self.conv(x1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 128, 32, 32)
