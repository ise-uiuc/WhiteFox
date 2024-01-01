
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input1, input3)
        t3 = torch.mm(input2, input1)
        t4 = torch.mm(input3, input1)
        t5 = torch.mm(input2, input2)
        t6 = torch.mm(input2, input3)
        t7 = torch.mm(input3, input3)
        t8 = t1 + t5
        t9 = t2 + t6
        t10 = t3 + t7
        return t8 - t10
# Inputs to the model
input1 = torch.randn(16, 64)
input2 = torch.randn(16, 64)
input3 = torch.randn(16, 64)
