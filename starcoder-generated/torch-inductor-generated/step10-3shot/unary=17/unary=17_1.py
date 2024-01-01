
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose0 = torch.nn.ConvTranspose2d(256, 256, 11, padding=0, stride=2)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(256, 3, 5, padding=0, stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose0(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_transpose1(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 256, 12, 12)
