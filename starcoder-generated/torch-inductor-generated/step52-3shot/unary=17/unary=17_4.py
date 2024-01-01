
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 1, 5)
        self.conv_transpose_1 = torch.nn.ConvTranspose1d(1, 2, kernel_size=5, stride=2, padding=1, output_padding=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose1d(2, 1, kernel_size=4, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose_1(v1)
        v3 = torch.relu(v2)
        v4 = self.conv_transpose_2(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 10)
