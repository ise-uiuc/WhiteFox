
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5, input6, input7, input8):
        t1 = torch.mm(input1, input4)
        t2 = torch.mm(input2, input3)
        t1 = torch.mm(input5, input3)
        t3 = torch.mm(input1, input4)
        t4 = torch.mm(input2, input3)
        t5 = torch.mm(input5, input3)
        t6 = torch.mm(input1, input4)
        t7 = torch.mm(input2, input3)
        t8 = torch.mm(input5, input3)
        return t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8
# Inputs to the model
input1 = torch.randn(4, 4)
input2 = torch.randn(4, 4)
input3 = torch.randn(4, 4)
input4 = torch.randn(4, 4)
input5 = torch.randn(5, 8)
input6 = torch.randn(5, 8)
input7 = torch.randn(4, 8)
input8 = torch.randn(4, 8)
