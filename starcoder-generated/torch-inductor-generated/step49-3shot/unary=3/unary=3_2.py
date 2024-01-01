
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, (5, 5), stride=(1, 1), padding=(2, 2))
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(64, 64, (5, 5), stride=(1, 1), padding=(2, 2))
        self.relu2 = torch.nn.ReLU()
    def forward(self, x: torch.Tensor):
        x1 = self.conv1(x)
        x2 = self.relu1(x1)
        x3 = self.conv2(x2)
        x4 = self.relu2(x3)
        return x4
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
