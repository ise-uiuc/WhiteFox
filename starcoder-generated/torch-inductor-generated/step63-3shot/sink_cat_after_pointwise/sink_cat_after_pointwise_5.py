
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.repeat(5, 1)
        return torch.cat((x, y), dim=0).relu()
# Inputs to the model
x = torch.randn(1, 3, 4)
