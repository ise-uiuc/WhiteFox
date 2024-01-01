
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, padding=2),
            torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Conv2d(32, 32, 3, dilation=2, padding=2),
            torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 16, 3, dilation=2, padding=2),
            torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(
            torch.nn.AvgPool2d(4, 4),
            torch.nn.Conv2d(16, 16, 3, dilation=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3, dilation=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(4, 4),
            torch.nn.Conv2d(16, 16, 1))
    def forward(self, x):
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        x = torch.tanh(x)
        x = self.layer3(x)
        x = torch.tanh(x)
        x = self.layer4(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
input = torch.randn(1, 3, 38, 38)
