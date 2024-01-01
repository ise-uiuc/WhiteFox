
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 4, 3, stride=2, padding=1)
        self.relu = torch.nn.ReLU()
        self.avg_pool2d = torch.nn.AvgPool2d(kernel_size=(10))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = self.relu(v2)
        v5 = self.avg_pool2d(v3)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
