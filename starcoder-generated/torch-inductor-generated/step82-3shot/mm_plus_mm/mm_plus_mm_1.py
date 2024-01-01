
class Model1(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = torch.mm(input1, input3)
        t4 = t1 + t2
        t5 = t1 + t3
        return t4*t5
class Model2(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = torch.mm(input1, input3)
        t4 = t1 + t2
        t5 = t1 + t3
        t6 = (t4 + t5)/2
        return t6
# Inputs to the model
input1 = torch.randn(6, 6)
input2 = torch.randn(6, 6)
input3 = torch.randn(6, 4)
input4 = torch.randn(4, 6)
