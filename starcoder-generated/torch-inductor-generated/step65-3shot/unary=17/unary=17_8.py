
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(8, 8, 1, padding=0, stride=1)
        self.conv_transpose1 = torch.nn.ConvTranspose1d(8, 8, 2, padding=0, stride=2, output_padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose1d(8, 8, 1, padding=0, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_transpose1(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv_transpose2(v4)
        v6 = torch.relu(v5)
        v7 = torch.sigmoid(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 8, 12)
