
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(2, out_channels=2, kernel_size=5, stride=3, padding=3)
        self.soft_relu = torch.nn.Softsign()
    
    def forward(self, x1):
        v1 = self.conv2()
        v2 = self.soft_relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 16, 16)
