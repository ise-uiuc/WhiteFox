
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_13 = torch.nn.ConvTranspose2d(1, 1, kernel_size=[[10.0, 6.0, 11.0], [17.0, 5.0, 2.0]], stride=[[1.0, 1.0], [1.0, 1.0]], padding=[[0.0, 0.0], [0.0, 0.0]])
    def forward(self, x1):
        v1 = self.conv_transpose_13(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(2, 1, 13, 11)
