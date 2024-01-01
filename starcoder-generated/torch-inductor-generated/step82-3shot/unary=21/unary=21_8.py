
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution2d = torch.nn.Conv2d(2, 5, 1, stride=1, padding=0)
    def forward(self, x):
        v1 = self.convolution2d(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 2, 3, 3)
