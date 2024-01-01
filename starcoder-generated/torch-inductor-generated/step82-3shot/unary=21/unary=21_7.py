
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution2d = torch.nn.Conv2d(3, 32, 3, stride=2)
        self.convolutiontranspose2d = torch.nn.ConvTranspose2d(32, 8, 2, stride=2)
    def forward(self, x):
        v1 = self.convolution2d(x)
        v2 = torch.tanh(v1)
        v3 = self.convolutiontranspose2d(v2)
        return v3
# Inputs to the model
x = torch.randn(32, 3, 32, 32)
