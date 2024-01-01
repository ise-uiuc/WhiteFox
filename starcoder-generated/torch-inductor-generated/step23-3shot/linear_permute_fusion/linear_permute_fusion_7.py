
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v1 = v1.permute(1, 0, 2, 3)
        return v1.permute(3, 2, 0, 1)
# Inputs to the model
x1 = torch.randn(3, 3, 2, 2, device='cpu')
