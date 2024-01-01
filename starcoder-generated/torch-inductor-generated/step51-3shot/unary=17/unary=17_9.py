
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, kernel_size=(3, 3), stride=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 4, kernel_size=(3, 3), stride=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_transpose(v2)
        v4 = torch.relu(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
