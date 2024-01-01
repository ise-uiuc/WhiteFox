
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.cat([x, x], dim=1)
        y = torch.cat([y, y], dim=-1)
        y = torch.cat([y, y], dim=0)
        y = torch.tanh(y.view(-1))
        return y.view(2, 2) if y.shape!= (4, 2) else y.view(2, 2)
# Inputs to the model
x = torch.randn(3, 2, 2)
