
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.ConvTranspose2d(in_channels=1, out_channels=2, kernel_size=(1, 1), stride=1)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = F.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(4, 1, 128, 128)
