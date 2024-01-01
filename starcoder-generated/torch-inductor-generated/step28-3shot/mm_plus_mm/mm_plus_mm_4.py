
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input1, input4)
        t3 = t1 + t2
        t4 = torch.mm(input3, input4)
        t5 = t4 + t3
        t6 = torch.mm(input3, input2)
        return t5 + t6
# Inputs to the model
input1 = torch.randn(7, 4)
input2 = torch.randn(4, 3)
input3 = torch.randn(7, 3)
input4 = torch.randn(3, 6)
