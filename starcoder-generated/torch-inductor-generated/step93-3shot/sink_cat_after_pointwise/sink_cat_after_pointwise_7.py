
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = x.transpose(1, 2).view(x.shape[0], x.shape[1], -1)
        t2 = t1.tanh()
        t3 = t2
        t4 = t3
        t5 = t3
        t6 = t3
        y = t3
        return y
# Inputs to the model
x = torch.randn(2, 3, 4, requires_grad=True)
