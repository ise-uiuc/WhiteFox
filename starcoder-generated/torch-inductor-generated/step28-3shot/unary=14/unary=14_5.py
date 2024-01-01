
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = torch.nn.Conv2d(2, 2, kernel_size=(2, 9), stride=(1, 7), padding=(0, 1))
    def forward(self, x1):
        v1 = self.conv2d_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
