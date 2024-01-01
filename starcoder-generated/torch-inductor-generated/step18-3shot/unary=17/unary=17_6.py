
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 32, 3, padding=0, stride=2)
    def forward(self, x):
        v1 = self.conv_transpose(x)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 16, 16)
