
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t2 = torch.mm(input3, input4)
        t3 = torch.mm(input3, input2)
        t4 = torch.mm(input4, input3)
        t5 = torch.mm(input3, input1)
        t6 = torch.mm(input4, input2)
        t7 = torch.mm(input1, input3)
        t8 = torch.mm(input2, input4)
        t9 = t2 + t3 + t4
        t10 = t5 + t6 + t7
        t11 = t8 + t9 + t10
        return t11
# Inputs to the model
input1 = torch.randn(2048, 2048)
input2 = torch.randn(2048, 1024)
input3 = torch.randn(2048, 256)
input4 = torch.randn(1024, 768)
