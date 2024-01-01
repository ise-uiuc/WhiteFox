
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 24, 8, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1.transpose(dim0=1, dim1=2)
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 1024, 1024)
