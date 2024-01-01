
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        t1 = torch.mm(input, input)
        with torch.no_grad():
            t2 = torch.mm(input, input)
        t3 = torch.mm(input, input)
        t4 = t3 + t1
        t5 = torch.mm(input, input)
        t6 = t5 + t4
        t7 = t6 + t2
        return t7
# Inputs to the model
input = torch.randn(4, 4)
