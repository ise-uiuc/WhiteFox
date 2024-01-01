
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 32, 5, stride=1, padding=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_transpose(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 30, 30)
