
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 16, 3, stride=2, padding=1)
    def forward(self, x):
        x = torch.tanh(self.conv(x))
        return x.squeeze(3)
# Inputs to the model
x = torch.randn(1, 8, imgHeight, imgWidth) # change the input tensor shape here
