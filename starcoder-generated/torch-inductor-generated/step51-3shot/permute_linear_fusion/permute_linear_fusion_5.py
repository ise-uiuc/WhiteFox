
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv2d = torch.nn.Conv2d(2, 2, (2, 2))
    def forward(self, x1):
        v = x1.permute(0, 2, 1, 3)
        y = v + torch.randn(1, 2, 2, 2)
        y = y.view([1, 2, 4])
        y = y[:, :, 0:2]
        return y
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
