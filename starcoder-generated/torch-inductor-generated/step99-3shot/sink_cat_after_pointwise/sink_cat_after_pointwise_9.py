
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x
        for i in range(3):
            x = y.view(-1, 2) * y.view(-1, 2)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
