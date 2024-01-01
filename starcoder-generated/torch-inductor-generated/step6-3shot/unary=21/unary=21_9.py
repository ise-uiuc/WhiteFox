
class ModelTanh(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 128, (3, 3), padding = (1, 1), stride = (1, 1), dilation = (2, 2))
        self.conv2 = torch.nn.Conv2d(128, 1, (1, 1))
    def forward(self, x) -> torch.Tensor:
        t1 = torch.nn.ReLU()(self.conv1(x))
        t2 = self.conv2(t1)
        t3 = torch.tanh(t2)
        return t3
# Inputs to the model
input = torch.randn(32, 4, 128, 128)
