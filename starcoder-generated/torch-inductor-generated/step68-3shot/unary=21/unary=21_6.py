
class ModelTanh(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        v1 = torch.tanh(x1)
        return x1
# Inputs to the model
x = torch.rand(1, 1, 47, 63)
