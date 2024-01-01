
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.zeros(1, 3, 10, 10)
        x = x + 4
        return x
# Inputs to the model
x1 = torch.ones((1, 3, 100, 100))
