
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(1, 512, 1, 1)
        self.conv2d = torch.nn.Conv2d(512, 1, 1, 1)
    def forward(self, x0):
        v0 = F.relu(x0)
        v1 = self.conv_transpose2d(v0)
        v10 = self.conv2d(v1)
        v3 = torch.sigmoid(v10)
        return v3
# Inputs to the model
x0 = torch.randn(1, 1, 5, 5)
