
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = x1.view(18, 0)
        x3 = x2.reshape(1, 18)
        x4 = x3.reshape(0)
        return x4
# Inputs to the model
x1 = torch.randn(1, )
