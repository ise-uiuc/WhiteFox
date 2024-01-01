
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x], dim=0)
        y = y.view(y.dim(), -1)
        if y.shape!= (6, 6):
            y = y + y
        return y.tanh() 
# Inputs to the model
x = torch.randn(2, 3)
