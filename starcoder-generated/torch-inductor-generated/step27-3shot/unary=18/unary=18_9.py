
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(1, 4, kernel_size=(2, 4), stride=(2, 2), padding=(3, 2), dilation=(4, 4))
    def forward(self, x1):
        v1 = self.conv2(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
