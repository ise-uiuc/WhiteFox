
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x):
        v1 = torch.mm(x, x)
        v2 = torch.mm(x, x)
        v3 = v1 * v2
        return v3
# Inputs to the model
x = torch.randn(5, 5)
