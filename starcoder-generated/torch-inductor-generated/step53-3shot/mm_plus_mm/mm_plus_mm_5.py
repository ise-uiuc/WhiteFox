
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input1, input2)
        t3 = torch.mm(input1, input2)
        t4 = torch.mm(input1, input2)
        t5 = torch.mm(input1, input2)
        t6 = torch.mm(input1, input2)
        t7 = torch.mm(input1, input2)
        t8 = torch.mm(input1, input2)
        t9 = torch.mm(input1, input2)
        return t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9
# Inputs to the model
input1 = torch.randn(16, 16)
input2 = torch.randn(16, 16)
input3 = torch.randn(16, 16)
input4 = torch.randn(16, 16)
