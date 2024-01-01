
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(in_channels=5, out_channels=6, kernel_size=(4, 2), stride=(3, 1),
                                                        padding=(2, 0))
        self.conv_transpose2 = torch.nn.ConvTranspose2d(in_channels=6, out_channels=4, kernel_size=(4, 3), stride=(2, 1),
                                                        padding=(0, 2))
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_transpose2(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 5, 8, 10, dtype=torch.float32)
