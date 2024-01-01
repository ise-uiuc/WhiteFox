
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        shape = x.shape
        x = torch.cat((x, x), dim=0)
        shape_list = list(x.shape)
        shape_list[0] //= 2
        x = x.reshape(1, *shape_list)
        if x.shape == torch.Size([2, 3, 2, 4, 5]):   
            x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4, 5)
