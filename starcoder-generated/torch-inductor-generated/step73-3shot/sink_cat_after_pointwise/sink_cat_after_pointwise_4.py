
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        shape_list = list(x.shape)
        shape_list[0] = -1
        x = torch.cat([x, x], dim=0)
        x = x.view(*shape_list)
        x = torch.relu(x)
        del shape_list
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
