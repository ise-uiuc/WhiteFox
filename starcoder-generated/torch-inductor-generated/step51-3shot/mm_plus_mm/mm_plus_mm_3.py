
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        t1 = torch.mm(input, input)
        t1c = t1.clone().detach()
        t2 = torch.mm(t1c, t1c)
        return t2
# Inputs to the model
input = torch.randn(100, 100)
