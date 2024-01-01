
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x1):
        t1 = torch.tanh(x1)
        t2 = torch.tanh(t1)
        t3 = torch.tanh(t2)
        t4 = torch.tanh(t3)
        t5 = torch.tanh(t4)
        t6 = torch.tanh(t5)
        return t6
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
