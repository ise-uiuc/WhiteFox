
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, (3, 3), (1, 1), (1, 1)),
            torch.nn.ConvTranspose2d(32, 64, kernel_size=4, stride=4),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2))
    def forward(self, x1):
        v1 = self.layers(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 4, 24, 24)
