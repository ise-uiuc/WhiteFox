
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        if (y.shape[1]!=  8):
            shape_list = list(y.shape)
            shape_list[1] = -1
            self.shape_list = shape_list
            y = y.view(*shape_list)
            del self.shape_list
        else:
            shape_list = list(y.shape)
            shape_list[1] = -1
            self.shape_list = shape_list
            y = y.view(*shape_list)
            del self.shape_list
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
