
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = torch.empty((10,))
        v1.fill_(0.)
        v2 = torch.empty((2,))
        v2.fill_(0.)
        v2.add_(1000000000000.)
        v3 = torch.empty((4,))
        v3.fill_(0.)
        v3.add_(1000000000000)
        v4 = v1 * v2 * v3
        return torch.cat([v4, v4], 0)
# Inputs to the model
x = torch.Tensor(10)
