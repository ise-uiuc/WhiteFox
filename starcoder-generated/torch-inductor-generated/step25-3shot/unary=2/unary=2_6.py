
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(33, 8, kernel_size=3, padding=1, stride=1)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(1, 23, kernel_size=3, padding=1, stride=1)
        self.relu2 = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.relu1(v1)
        v3 = self.conv_transpose2(x1)
        v4 = self.relu2(v3)
        v5 = v1 + v4
        return v5
# Inputs to the model
x1 = torch.randn(3, 33, 2, 2)
