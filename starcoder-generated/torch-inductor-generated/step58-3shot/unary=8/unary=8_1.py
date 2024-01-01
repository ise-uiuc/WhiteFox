
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.average_pool = torch.nn.AvgPool2d(stride=4)
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 12, kernel_size=(2, 2), padding=(3, 2), dilation=5)
        self.relu = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = self.average_pool(x1)
        v2 = self.conv_transpose(v1)
        v3 = self.relu(v2)
        v4 = v3
        v5 = v3
        v6 = v4 + v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
