
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(2, 3, 1, stride=1, padding=1, output_padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(3, 3, 3, stride=3, padding=2, output_padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 7, 7)
