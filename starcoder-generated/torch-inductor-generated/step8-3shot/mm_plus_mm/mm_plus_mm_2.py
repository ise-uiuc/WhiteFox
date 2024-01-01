
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5, input6):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input1, input4)
        t3 = torch.mm(input1, input6)
        t4 = torch.mm(input3, input2)
        t5 = torch.mm(input3, input4)
        t6 = torch.mm(input3, input6)
        t7 = t1 + t2
        t8 = t3 + t4
        t9 = t5 + t6
        t10 = t7 + t8
        t11 = t9 + t10
        t12 = t11 + t7
        return t12
# Inputs to the model
input1 = torch.randn(4, 3)
input2 = torch.randn(3, 5)
input3 = torch.randn(4, 3)
input4 = torch.randn(3, 5)
input5 = torch.randn(4, 3)
input6 = torch.randn(3, 5)
