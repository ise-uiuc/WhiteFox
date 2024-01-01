
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.ConvTranspose2d(12, 3, 1, stride=1)
        self.conv_3 = torch.nn.ConvTranspose2d(3, 12, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_3(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 12, 4, 4)
