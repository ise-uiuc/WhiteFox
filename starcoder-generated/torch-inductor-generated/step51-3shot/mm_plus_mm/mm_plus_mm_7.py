
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        t1 = torch.mm(input, input)
        t1 = t1 + t1
        t2 = t1 + t1
        t2 = t2 + t1
        return t2
# Inputs to the model
input = torch.randn(1, 100)
