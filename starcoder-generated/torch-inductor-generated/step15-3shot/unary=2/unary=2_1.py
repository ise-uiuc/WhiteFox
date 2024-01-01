
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(1, 24, kernel_size=(3, 3), stride=(3, 3), padding=1)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=(3, 3), stride=(3,3), padding=2)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.avg_pool(v1)
        return v2
# Inputs to the model
x1 = torch.randn(5, 1, 2, 6)
