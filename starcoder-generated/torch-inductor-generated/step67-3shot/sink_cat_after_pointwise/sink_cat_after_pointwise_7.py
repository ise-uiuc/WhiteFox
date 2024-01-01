
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        shape1 = [1] * len(x.shape)
        shape0 = list(z.shape)
        shape0[0] = -1
        y1 = list(x.shape)
        y1[0] = -1
        x1 = x.view(shape1)
        x2 = x1.view(*shape0)
        y2 = x2.tanh()
        return y2
# Inputs to the model
x = torch.randn(3, requires_grad=True)
