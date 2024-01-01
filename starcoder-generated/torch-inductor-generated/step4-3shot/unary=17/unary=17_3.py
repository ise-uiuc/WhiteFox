
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 8, 1, stride=1, padding=4)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.nn.functional.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
