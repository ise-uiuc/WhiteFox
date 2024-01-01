
class ConvModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3,3,2)
    def forward(self, x, y):
        x1 = F.dropout2d(x, p=0.7)
        x2 = y + 1
        return self.conv(x1) + x2
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
