
class Module_15(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        y1 = torch.split(x, 1, 3)
        y2 = operator.getitem(y1, 1)
        y6 = torch.cat((y2, y2), 3)
        return y6

# Initializing the model
m = Module_15()

# Inputs to the model
x = torch.randn(1, 1, 3, 64)
