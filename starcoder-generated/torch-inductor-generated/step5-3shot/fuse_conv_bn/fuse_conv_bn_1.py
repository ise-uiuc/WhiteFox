
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(5)
        self.bn2 = torch.nn.BatchNorm2d(3)
    def forward(self, input_t):
        # We cannot fuse this pattern due to `bn2` being in training mode
        y = self.bn1(input_t)
        z = self.bn2(y)
        q = self.bn1(z)
        return q
# Inputs to the model
x = torch.randn(1, 5, 5, 5)
