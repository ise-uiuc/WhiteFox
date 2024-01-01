
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.cat([x, x, x, x], dim=1)
        t2 = t1.view(x.shape[0], x.shape[1], -1)
        t3 = t2.tanh()
        t4 = t3.relu()
        x = t4
        return x
# Inputs to the model
x = torch.randn(1, 3, 4)
