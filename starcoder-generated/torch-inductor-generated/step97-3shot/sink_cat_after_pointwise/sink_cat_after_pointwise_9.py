
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        z = y.view(-1, 2 * (1 * 4))
        z = z @ torch.randn(2 * 3, 5)
        z = z.relu()
        return z
# Inputs to the model
x = torch.randn(2, 3, 4)
