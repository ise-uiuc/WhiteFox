
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_791 = torch.nn.ConvTranspose2d(110, 166, 3, stride=(1, 2), padding=(1, 2))
    def forward(self, x1):
        v1 = self.conv_transpose_791(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = torch.nn.functional.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 110, 35, 38)
