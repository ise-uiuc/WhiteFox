
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(2, 64, 3, stride=(2, 2, 1), padding=(2, 0, 0))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 32, 16, 64)
