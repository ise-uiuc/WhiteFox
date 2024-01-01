
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x, x], dim=1)
        new_shape = y.shape[:-1] + (-1,)
        y = y.reshape(*new_shape)
        x = y.relu()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
