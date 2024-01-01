
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=1),
            torch.nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=1),
        )
    def forward(self, x1):
        v1 = self.layer1(x1)
        v2 = v1.add(3.0)
        return torch.clamp_max(v2, 6.0)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
