
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5, input6):
        t1 = torch.mm(input1, input3)
        t2 = torch.mm(input2, input4)
        t3 = torch.mm(input5, input6)
        t4 = torch.mm(input1, input3)
        t5 = torch.mm(input2, input4)
        t6 = torch.mm(input5, input6)
        t7 = torch.mm(input1, input6)
        t8 = torch.mm(input2, input5)
        t9 = torch.mm(input3, input6)
        t10 = torch.mm(input1, input3)
        t11 = torch.mm(input2, input4)
        t12 = torch.mm(input5, input6)
        t13 = torch.mm(input3, input5)
        t14 = torch.mm(input1, input2)
        t15 = torch.mm(input4, input6)
        t16 = t1 + t2 + t3
        t17 = t4 + t5 + t6
        t18 = t7 + t8 + t9
        t19 = t10 + t11 + t12
        t20 = t13 + t14 + t15
        t21 = t16 + t17
        t22 = t18 + t19
        t23 = torch.mm(t21, t22)
        return t23
# Inputs to the model
input1 = torch.randn(3, 3)
input2 = torch.randn(3, 3)
input3 = torch.randn(3, 3)
input4 = torch.randn(3, 3)
input5 = torch.randn(3, 3)
input6 = torch.randn(3, 3)
