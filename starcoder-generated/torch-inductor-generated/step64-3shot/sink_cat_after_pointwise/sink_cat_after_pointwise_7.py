
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()       
    def forward(self, x):
        y = torch.cat((x, x, x), dim=1)
        y = y.view(y.shape[0], -1)
        t1 = torch.relu(y)
        return t1 if tuple(y.shape) == (1, 3) else t1.view(tuple(y.shape)[0], -1)
# Inputs to the model
x = torch.randn(2, 3, 4)
