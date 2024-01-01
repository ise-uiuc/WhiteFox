
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 32, 5, padding=2, stride=1)
    def forward(self, x1):
        v15 = self.conv(x1)
        v15 = F.relu(v15)
        v16 = torch.sigmoid(v15)
        return v16
# Inputs to the model
x1 = torch.randn(1, 3, 360, 360)
