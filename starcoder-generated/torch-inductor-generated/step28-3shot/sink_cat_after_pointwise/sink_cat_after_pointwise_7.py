
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x], dim=-1)
        x = y.split(2, dim=-2)
        return x
# Inputs to the model
x = torch.randn(*torch.randint(2, 10, (6,)))
