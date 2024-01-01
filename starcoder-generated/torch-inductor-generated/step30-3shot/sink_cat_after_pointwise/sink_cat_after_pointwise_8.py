
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.cat((x, x), dim=1)
        t2 = t1.view(x.shape[0], -1)
        t3 = t2.tanh()
        t4 = t3.view(x.shape[0], -1).tanh()
        x = t4
        z = x + 2
        y = z * 3
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
