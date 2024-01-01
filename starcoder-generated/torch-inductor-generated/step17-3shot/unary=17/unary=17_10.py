
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.ConvTranspose1d(32, 64, 3, padding=1, stride=1, output_padding=1)
    def forward(self, x1):
        v1 = self.conv1d(x1)
        v2 = F.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(7, 32, 512)
