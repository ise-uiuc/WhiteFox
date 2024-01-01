
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        y = torch.cat((x, y), dim=1)
        y = y.view(y.shape[0], -1)
        y = y.add(-1)
        # no pointwise op here
        return y
# Inputs to the model
x = torch.rand(2, 3, 4)
y = torch.rand(2, 2, 4)
