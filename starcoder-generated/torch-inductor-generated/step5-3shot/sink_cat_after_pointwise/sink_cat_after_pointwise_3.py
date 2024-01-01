
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        x = y.view(y.shape[0], -1)
        y = x.view(y.shape[0], -1).relu()
        x = y.view(-1, y.shape[1]).relu()
        y = x.view(y.type(torch.BoolTensor))
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
