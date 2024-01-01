
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5, input6):
        t1 = torch.mm(input1, input4)
        t2 = torch.mm(input2, input1)
        t3 = torch.mm(input3, input2)
        t4 = torch.mm(input4, input3)
        t5 = torch.mm(input5, input5)
        t6 = torch.mm(input6, input6)
        return t1 + t2 + t3 + t4 + t5 + t6
# Inputs to the model
input1 = torch.randn(9, 9)
input2 = torch.randn(9, 9)
input3 = torch.randn(9, 9)
input4 = torch.randn(9, 9)
input5 = torch.randn(9, 9)
input6 = torch.randn(9, 9)
