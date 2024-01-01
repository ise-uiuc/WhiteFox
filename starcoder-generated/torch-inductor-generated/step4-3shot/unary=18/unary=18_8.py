
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(37, 68, (17, 17), stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(68, 56, (17, 17), stride=2, padding=9, dilation=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(56, 53, 20, 29, 10, output_padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(53, 48, (7, 7), stride=3, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(48, 3, (17, 17), stride=1, padding=1)
        )
    def forward(self, x1):
        v1 = self.layers(x1)
        return nn.Sigmoid()(v1)
# Inputs to the model
x1 = torch.randn(1, 37, 1, 1)
