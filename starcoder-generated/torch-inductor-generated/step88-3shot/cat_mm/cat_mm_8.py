
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, h):
        v = h[0][0]
        v = h[-1][0]
        v = h[1][0]
        for loopVar1 in range(100):
            v = h[0][0]
            v = h[-1][0]
            v = h[1][0]
        return h[0][0]
# Inputs to the model
h = [[[1., 2., 3.]]]
