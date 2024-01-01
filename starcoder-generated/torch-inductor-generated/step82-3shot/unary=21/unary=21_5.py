
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution = torch.nn.Conv2d(3, 22, 1, stride=1, padding=0)
        self.leakyrelu = torch.nn.LeakyReLU(0.0)
    def forward(self, x):
        v1 = self.convolution(x)
        v2 = self.leakyrelu(v1)
        return v2
# Inputs to the model
tensor = torch.randn(1, 3, 16, 16)
