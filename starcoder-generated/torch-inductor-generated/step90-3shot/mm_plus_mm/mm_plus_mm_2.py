
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input3, input4)
        t2 = torch.mm(input1, input2)
        t3 = t1.mm(input4)
        t4 = torch.mm(input2, input4)
        t5 = torch.mm(input3, input2)
        t6 = torch.mm(input2, input4)
        t7 = t2 + t3
        t8 = t4 + t5
        t9 = t7 + t6
        t10 = t2 + t8
        t11 = t3 + t9
        t12 = t10.mm(t11)
        t13 = t7 + t10
        t14 = t4 + t11
        t15 = t5 + t13
        t16 = t6 + t14
        return t12.mm(t16)
# Inputs to the model
x1 = torch.randn(6, 6)
x2 = torch.randn(6, 6)
x3 = torch.randn(6, 6)
x4 = torch.randn(6, 6)
