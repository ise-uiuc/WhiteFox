
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(2),
            torch.nn.Sigmoid(),
            torch.nn.Conv2d(2, 4, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.Sigmoid(),
        )
    def forward(self, x1):
        v1 = self.convs(x1)
        v2 = torch.nn.functional.interpolate(v1, scale_factor=2.5, mode='bicubic')
        return v2
# Inputs to the model
x1 = torch.randn(1, 32, 143, 143)
