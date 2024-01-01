
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2=None, other=None):
        if other!= None:
            v1 = x1 + other
        else:
            x2 = x1 + x2
            v1 = x2
        if x2 == None:
            x2 = torch.randn(v1.shape)
        v2 = x2 + v1
        return v2
# Inputs to the model
x1 = torch.randn(1, 5, 32, 32)
