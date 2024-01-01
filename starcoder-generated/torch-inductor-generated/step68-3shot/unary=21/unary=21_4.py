
class ModelTanh(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 7, stride=2, padding=0)
    def forward(self,x) -> torch.Tensor:
        x = self.conv1(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 32, 32)
