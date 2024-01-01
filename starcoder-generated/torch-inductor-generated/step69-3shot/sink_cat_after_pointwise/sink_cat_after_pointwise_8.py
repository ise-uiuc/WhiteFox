
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        shape_list = list(x.shape)
        shape_list_a = list(shape_list)
        shape_list_a[0] = -1
        shape_list_b = list(shape_list)
        shape_list_b[1] = -1
        y = torch.cat([x, x, x], dim=1)
        y = y.view(*shape_list_a)
        x = x.relu()
        del shape_list, shape_list_a, shape_list_b
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
