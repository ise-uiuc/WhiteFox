
class ModelTanh(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 1, (1, 7), stride=(1, 1), bias=False)
        self.conv_2 = torch.nn.Conv2d(1, 1, (5, 1), stride=(1, 1), bias=False)
        self.conv_3 = torch.nn.Conv2d(1, 1, (4, 4), stride=(1, 1), bias=False)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        x = self.tanh(self.conv_1(x))
        x = self.tanh(self.conv_2(x))
        x = self.tanh(self.conv_3(x))
        return x
# Inputs to the model
x = torch.randn(1, 1, 41, 41)
