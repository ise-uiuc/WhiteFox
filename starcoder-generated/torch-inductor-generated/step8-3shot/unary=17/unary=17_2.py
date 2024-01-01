
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose0 = torch.nn.ConvTranspose2d(1, 64, 9, padding=0, stride=2)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(64, 32, 1, padding=0,stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose0(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_transpose1(v2)
        v4 = torch.max_pool2d(v3, 2, stride=1)
        return v4
# Inputs to the model
x1 = torch.randn(2, 1, 32, 32)
