
class ModelTanh(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 1, 1, stride=1, padding=0)
        self.tanh1 = torch.nn.Tanh()
    def forward(self, x) -> torch.Tensor:
        x = x.view(-1, 9, 11, 3)
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x) + self.conv3(x)) - self.conv2(x)
        x = torch.tanh(self.conv4(x))
        return x
# Inputs to the model
x = torch.rand(1, 1, 10, 77)
