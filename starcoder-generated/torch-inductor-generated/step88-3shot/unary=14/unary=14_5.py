
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = torch.nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True)
        self.conv_transpose_20 = torch.nn.ConvTranspose2d(3, 3, 3, stride=1, padding=0, bias=True)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        v1 = self.avg_pool(x1)
        v2 = self.relu(v1)
        v3 = self.conv_transpose_20(v2)
        v4 = torch.sigmoid(v3)
        v5 = v3 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 2, 3, requires_grad=True)
