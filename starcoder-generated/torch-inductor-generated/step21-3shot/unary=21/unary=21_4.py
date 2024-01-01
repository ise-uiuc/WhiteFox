
class ModelTanh(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 2, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 4, stride=4, padding=2)
    def forward(self, x) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 128, 128)
