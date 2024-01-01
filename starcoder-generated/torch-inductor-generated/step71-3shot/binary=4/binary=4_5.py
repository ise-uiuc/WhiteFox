
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        b1 = self.batch_norm(x)
        b2 = b1 + other
        return b2

# Initializing the model
__model__ = Model()

# Inputs to the model
__input__ = torch.randn(1, 3, 64, 64)
