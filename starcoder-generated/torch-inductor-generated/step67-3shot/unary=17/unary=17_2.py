
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(1, 2, kernel_size=2)
    def forward(self, x):
        x = self.conv(x)
        return F.relu(x)
# Inputs to the model
x = torch.randn(1, 1, 2, 2)
