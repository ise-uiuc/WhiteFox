
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(3, 5, 1),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose1d(5, 7, 1),
        )
    def forward(self, x1):
        v1 = self.layers(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 13)
