
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        t1 = torch.randn(3, 3, requires_grad=True)
    def forward(self, x1, inp):
        v1 = self.linear(x1)
        input1 = v1
        v2 = v1.reshape(1, 9)
        v3 = input1.reshape(9, 1) + v2
        v4 = torch.mm(v3, v3)
        return (v4 + inp, torch.mm(x1, v4).reshape(3, 3) + v4)
# Inputs to the model
x1 = torch.randn(3, 3)
inp = torch.randn(3, 3)
