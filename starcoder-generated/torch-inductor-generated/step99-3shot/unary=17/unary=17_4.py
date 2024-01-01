
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v1)
        return v2, v3
# Inputs to the model
x1 = torch.randn(1, 1, 4, 4)
