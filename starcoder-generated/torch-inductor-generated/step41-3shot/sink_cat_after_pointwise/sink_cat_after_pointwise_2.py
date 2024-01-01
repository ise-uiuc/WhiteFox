
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, x1):
        y = torch.cat([x, x1], dim=1)
        x = y.view(-1).div(2)
        y = torch.cat([x, x1], dim=1)
        x = y.view(-1).div(2)
        z = y.view(-1).tanh()
        x = z.div(2)
        y = torch.cat([x, x1], dim=1)
        y = y.view(-1).sigmoid()
        return y
# Inputs to the model
x = torch.randn(1, 2)
x1 = torch.randn(1, 3)
