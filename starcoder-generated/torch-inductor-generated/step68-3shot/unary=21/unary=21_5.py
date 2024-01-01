
class ModelTanh(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_3 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.tanh = torch.nn.Tanh()
    def forward(self,x) -> torch.Tensor:
        x = self.conv_1(x)
        y = torch.tanh(x)
        y = self.conv_2(y)
        y = torch.tanh(y)
        y = self.conv_3(y)
        y = torch.tanh(y)
        return y
# Inputs to the model
x = torch.rand(1, 3, 64, 64)
