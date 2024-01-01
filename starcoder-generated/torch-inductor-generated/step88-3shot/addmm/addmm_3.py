
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        inp = torch.randn(5, requires_grad=True)
        # Note that inputs are not all of the ones listed below.
        # Feel free to add different inputs to trigger the pattern.
        v1 = torch.mm(x, inp)
        v2 = torch.mm(x, v1)
        return v2
# Inputs to the model
x = torch.randn(3, 5)
