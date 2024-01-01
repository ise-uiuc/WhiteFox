
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, [0, 1], stride=1, padding=[[1, 2], [3, 4]])
        # This conv has kernel size of [0, 1] and padding of [[1, 2], [3, 4]]
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return (v1, v3)
# Inputs to the model
x1 = torch.randn(1, 1, 2, 3)
