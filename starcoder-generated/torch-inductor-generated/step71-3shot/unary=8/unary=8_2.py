
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 16, kernel_size=4, stride=2, padding=1)
        self.relu = torch.nn.ReLU6()
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v1 = self.relu(v1)
        v1 = self.avg_pool(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 15, 15)
