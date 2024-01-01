
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        t1 = torch.mm(input, input)
        t2 = t1 + t1
        t3 = torch.mm(input, input)
        t4 = t2 + t3
        t5 = t4 + t3
        return t5 
# Inputs to the model
input = torch.randn(100, 100)
