
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 4, (3, 1), stride=(2, 1), padding=(1, 0), output_padding=(1, 0))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
