
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t1 = torch.mm(input1, input3)
        t2 = torch.mm(input2, input3)
        t3 = t1 + t2
        t4 = torch.mm(input3, input3)
        t5 = torch.mm(input3, input3)
        t6 = torch.mm(input3, input3)
        t7 = torch.mm(input3, input3)
        t8 = torch.mm(input3, input3)
        t9 = torch.mm(input3, input3)
        t10 = torch.mm(input3, input3)
        t11 = t4 + t5 + t6 + t7 + t8 + t9 + t10
        return t3 + t11
# Inputs to the model
input1 = torch.randn(3, 3)
input2 = torch.randn(3, 3)
input3 = torch.randn(3, 3)
