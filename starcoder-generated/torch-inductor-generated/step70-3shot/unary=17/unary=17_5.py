
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 16, 3, stride=2, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1)
        self.pointwise_conv = torch.nn.Conv2d(8, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.relu(v1)
        v3 = self.conv_transpose2(v2)
        v4 = self.relu(v3)
        v5 = self.pointwise_conv(v4)
        return torch.tanh(v5)
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
