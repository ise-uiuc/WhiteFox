
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
        self.conv = torch.nn.ConvTranspose2d(1, 3, kernel_size=16)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.maxpool(x1)
        v2 = self.conv(v1)
        v3 = self.relu(v2)
        return v1, v3
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
