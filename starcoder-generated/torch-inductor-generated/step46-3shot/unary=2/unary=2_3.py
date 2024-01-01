
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(1, 11, (4, 1, 3), stride=1, padding=(3, 0, 1))
        self.avg_pool = torch.nn.AvgPool2d(3, stride=3)
        self.max_pool = torch.nn.MaxPool2d(3, stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.avg_pool(v1)
        v3 = self.max_pool(v2)
        return (0.2 * v3)
# Inputs to the model
x1 = torch.randn(1, 1, 4, 4, 4)
