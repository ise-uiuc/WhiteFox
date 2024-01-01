
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        shape_list = list(x.shape)
        shape_list[1] = -1
        self.shape_list = shape_list
        x = torch.cat([x, x, x], dim=1)
        x = x.relu()
        x = x.view(*shape_list)
        del self.shape_list
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
