
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t0 = torch.ones(10, 25, 14)
        t1 = torch.cat((t0, t0), dim=1)
        t2 = torch.cat((t1, t1), dim=0)
        t3 = t2.view(t2.shape[0], -1)
        y = torch.cat((t3, x), dim=1)
        return y.relu()
# Inputs to the model
x = torch.ones(20, 25, 14)
