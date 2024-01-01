
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 12, 1, stride=1)
        self.pool = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.conv_1(x)
        v2 = self.pool(v1)
        return v2
# Inputs to the model
x = torch.randn(64, 3, 224, 224)
