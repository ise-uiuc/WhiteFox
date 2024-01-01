
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=4, padding=0, dilation=3),
            torch.nn.Sigmoid(),
        )
    def forward(self, x1):
        v1 = self.model(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
