
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):

        # this would trigger if t1.size(0) is not a const

        t0 = torch.cat((x, x, x), dim=1)
        if t0.dim() < 2 or t0.size(0) < 2:
            return t0.view(t0.size(0), -1).tanh()
        else:
            return t0.view(t0.size(0), -1).tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)
