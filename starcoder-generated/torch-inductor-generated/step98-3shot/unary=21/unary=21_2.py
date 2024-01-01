
class ModelTanh(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 3, kernel_size=(2, 2), stride=(1, 1))
        self.tanh = torch.nn.Tanh()
    def forward(self, x) -> torch.Tensor:
        x1 = self.conv_1(x)
        x2 = self.tanh(x1)
        return x2
# Inputs to the model
input = torch.randn(1, 1, 18, 18)
