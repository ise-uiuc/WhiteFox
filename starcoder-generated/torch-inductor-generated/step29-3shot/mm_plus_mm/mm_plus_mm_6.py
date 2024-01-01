
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x):
        t1 = torch.mm(x, x)
        t2 = torch.mm(t1, x)
        t3 = torch.sin(x)
        return torch.mm(t3, t2) + torch.mm(t1, t3)
# Inputs to the model
input1 = torch.randn(6, 2)
