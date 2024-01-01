
class ModelTanh(torch.nn.Module):
    def __init__(self) -> None:
        super(ModelTanh, self).__init__()
        self.conv = torch.nn.Conv2d(240, 64, 1, 1, 0, 1, 1)
    def forward(self, x) -> torch.Tensor:
        t1 = self.conv(x)
        t2 = torch.tanh(t1)
        return t2
# Inputs to the model
input = torch.randn(1, 240, 400, 400)
