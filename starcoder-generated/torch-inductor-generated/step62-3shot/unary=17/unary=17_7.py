
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(in_channels=19, out_channels=5, kernel_size=(3, 3, 3))
        self.conv_transpose1 = torch.nn.ConvTranspose3d(5, 1, kernel_size=(2, 2, 2))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_transpose1(v2)
        v4 = torch.relu(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 19, 128, 128, 128)
