
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.ConvTranspose1d(1, 64, 5, padding=0, stride=1)
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 1, 128)
