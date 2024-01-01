
class ModelTanh(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 2, (2, 2), dilation=(2, 2), stride=(2, 2), groups=2)
    def forward(self, x) -> torch.Tensor:
        x1 = torch.tanh(self.conv1(x))
        x2 = torch.nn.ReLU()(x1)
        x3 = x2 * x1
        return x3
# Inputs to the model
x = torch.randn(1, 5, 30, 30)
