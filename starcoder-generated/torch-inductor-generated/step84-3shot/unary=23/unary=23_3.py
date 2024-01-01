
inputs = torch.FloatTensor(1, 2, 5, 7)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(1,4, 3, stride=2, padding=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 4, 3, stride=2, padding=3)
    def forward(self, x):
        v1 = self.conv2d(x)
        v2 = self.conv_transpose(v1)
        return v2
# Inputs to the model
x = torch.randn(inputs.shape)
