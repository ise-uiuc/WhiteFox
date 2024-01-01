
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = t1 + t2
        t4 = torch.mm(input5, input5)
        t5 = torch.mm(input1, input2)
        t6 = torch.mm(input3, input4)
        t7 = t5 + t6
        t8 = torch.mm(input5, input5)
        return t3 + t7 + t4 + t8
# Inputs to the model
input1 = torch.randn(32, 32)
input2 = torch.randn(32, 32)
input3 = torch.randn(32, 32)
input4 = torch.randn(32, 32)
input5 = torch.randn(32, 32)
