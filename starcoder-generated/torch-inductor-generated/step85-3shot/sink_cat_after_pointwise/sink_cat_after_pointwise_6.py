
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t0 = x.view(x.shape[0], 1, 1)
        t0.tanh()
        t1 = torch.cat((t0, t0, t0), dim=1)
        return t1.view(t1.shape[0], -1) if t1.shape[0] == 1 else t1.relu()
# Inputs to the model
x = torch.randn(1, 1, 1)
