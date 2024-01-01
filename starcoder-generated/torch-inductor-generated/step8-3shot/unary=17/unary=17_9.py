
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.avg_pool2d(x1, kernel_size=1, stride=1, padding=0)
        v2 = torch.squeeze(v1)
        v3 = torch.conv_transpose2d(v2, out_channels=8, kernel_size=3, stride=1, padding=1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
