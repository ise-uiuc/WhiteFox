
class ModelNoTanh(torch.nn.Module):
    def __init__(self):
        super(ModelNoTanh, self).__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
    def forward(self, x1):
        r1 = self.conv(x1)
        return r1
# Inputs to the model
x1 = torch.randn(10, 1, 16, 16)
