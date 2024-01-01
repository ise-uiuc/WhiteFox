
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x0):
        x2 = x0
        x2 = x2
        x1 = x2
        x1 = x1
        x0 = x1
        x0 = x0
        y = x0.squeeze(3).squeeze(3)
        return y


# Inputs to the model
x0 = torch.randn(9, 3, 1900, 256)
