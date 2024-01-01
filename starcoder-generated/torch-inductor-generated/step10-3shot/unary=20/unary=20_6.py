
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._conv1 = torch.nn.ConvTranspose2d((64,), (32,), (1, 1), (1, 1), 0, bias=False)
    def forward(self, x):
        v = self._conv1(x)
        return v
# Inputs to the model
x = torch.randn(1, 64, 112, 112)
