
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool_2d = torch.nn.AvgPool2d(3, stride=3)
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 4, 1, stride=3)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.relu(v1)
        v3 = self.avgpool_2d(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
