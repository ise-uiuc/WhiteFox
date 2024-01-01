
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 8, 3, stride=1, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(8, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        v3 = v2 + 1
        v4 = self.conv_transpose2(v3)
        v5 = F.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
x2 = torch.randn(1, 3, 7, 7)
