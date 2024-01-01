
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=(4,4), stride=2,padding=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=(4,4), stride=2,padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v2.transpose(3,2)
        v4 = self.conv_transpose_2(v3)
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
