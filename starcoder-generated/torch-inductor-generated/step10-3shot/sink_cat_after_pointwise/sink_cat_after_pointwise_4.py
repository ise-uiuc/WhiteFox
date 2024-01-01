
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = x.view(1, -1)
        t2 = x.view(2, -1)
        t3 = torch.cat((t1, t2), dim=0)
        t4 = t3.tanh()
        t5 = t4.relu()
        return t5
# Inputs to the model
x = torch.randn(2, 3, 4)
