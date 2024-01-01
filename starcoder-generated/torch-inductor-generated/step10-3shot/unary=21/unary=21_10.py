
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 7, stride=2, padding=2, bias=True, dilation=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 16, 5, stride=2, padding=0, bias=False),
            torch.nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)
# Inputs to the model
x = torch.randn(2, 3, 8, 8)
