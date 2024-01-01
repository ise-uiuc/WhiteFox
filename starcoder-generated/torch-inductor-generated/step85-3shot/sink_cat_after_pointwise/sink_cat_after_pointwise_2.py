
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t0 = x.view(x.shape[0], 1, 1)
        t1 = x.view(t1.shape[0], -1).tanh()
        t2 = torch.cat((t0, t1), dim=1)
        return t2.relu()
# Inputs to the model
x = torch.randn(1, 1, 1)
