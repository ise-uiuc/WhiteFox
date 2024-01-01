
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = x.view(4, 2, 2).sum(dim = 2).sum(dim=1)
        t1 = t1.view(-1, 4)
        t2 = torch.cat((t1, t1), dim=1)
        y, z = torch.log_softmax(t2, dim=1),  t2.cos()
        return y, z
# Inputs to the model
x = torch.randn(1, 8, 8)
