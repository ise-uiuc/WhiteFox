
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
        )
    def forward(self, x1):
        v1 = self.features(x1)
        v2 = v1.add_(3)
        v3 = v2.clamp(min=0, max=6)
        out = v3.div(6)
        return out
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
