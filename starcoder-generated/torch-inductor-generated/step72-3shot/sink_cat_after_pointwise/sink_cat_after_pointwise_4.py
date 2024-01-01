
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shape_list = list(torch.tensor(x.shape).numpy())
    def forward(self, x):
        y = torch.cat([x, x, x], dim=1)
        y = y.sigmoid()
        y = y.relu()
        y = y.view(*self.shape_list)
        return y
# Inputs to the model
x = torch.randn(2, 4)
