
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, input1):
        t1 = torch.mm(input1, input1)
        t2 = torch.mm(input1, input1)
        t3 = torch.mm(input1, input1)
        t4 = torch.mm(input1, input1)
        t5 = t1 + t2 + t3 + t4
        t6 = torch.mm(input1, input1)
        t7 = torch.mm(input1, input1)
        t8 = t6 + t7
        t9 = torch.mm(input1, input1)
        t10 = t6 + t9
        t11 = t7 + t10
        return t5 + t8 + t11
# Inputs to the model
input1 = torch.randn(1, 1)
