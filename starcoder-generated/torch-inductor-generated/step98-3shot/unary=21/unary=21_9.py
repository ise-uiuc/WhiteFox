
class ModelTanh(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2),
                                     dilation=(1, 1), groups=4, bias=True)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                     dilation=(1, 1), groups=1, bias=True)
        self.conv3 = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                     dilation=(1, 1), groups=1, bias=True)
        self.conv4 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                     dilation=(1, 1), groups=8, bias=True)

    def forward(self, x) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = torch.tanh(x2)
        x4 = self.conv3(x3)
        x5 = torch.tanh(x4)
        x6 = self.conv4(x5)
        x7 = torch.tanh(x6)
        return x7
# Inputs to the model
x = torch.randn(1, 4, 10, 10)
