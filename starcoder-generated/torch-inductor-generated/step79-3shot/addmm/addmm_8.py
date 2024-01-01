
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x0):
        s0 = torch.mm(x0[0][0], x0[0][1])
        return s0
# Inputs to the model
x0 = ((torch.randn(3, 3, 3), torch.randn(3, 3)),)
