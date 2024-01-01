
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6_ = torch.nn.ReLU6(inplace=True)
        self.conv_transpose2d = torch.nn.ConvTranspose2d(3, 4, 3, stride=2, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose2d(x1)
        v2 = self.relu6_(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
