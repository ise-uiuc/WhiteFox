
class ModelSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 6, 1, stride=1, padding=2, dilation=2)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.sigmoid(v1)
        return v2
# Inputs to the model
x = torch.randn(19, 2, 32, 34)
