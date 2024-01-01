
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(8, 64, 3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=True),
            torch.nn.Dropout2d(0.3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 1, 1, stride=1, padding=1, bias=True),
            torch.nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.model(x)
        return x
# Inputs to the model
x = torch.randn(1, 8, 32, 32)
