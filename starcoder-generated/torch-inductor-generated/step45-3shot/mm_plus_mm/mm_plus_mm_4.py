
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5, input6):
        t1 = torch.mm(input1, input5)
        t2 = torch.mm(input2, input6)
        t3 = t1 + t2
        t4 = torch.mm(input3, input2)
        t5 = torch.mm(input4, input3)
        t6 = t4 + t5
        t7 = t3 + t6
        return t7
# Inputs to the model
input1 = torch.randn(4, 4)
input2 = torch.randn(4, 4)
input3 = torch.randn(4, 4)
input4 = torch.randn(4, 4)
input5 = torch.randn(4, 4)
input6 = torch.randn(4, 4)

