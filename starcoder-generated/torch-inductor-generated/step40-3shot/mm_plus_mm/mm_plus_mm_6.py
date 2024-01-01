
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = t1 + t2
        t4 = torch.mm(input1, t3)
        t5 = torch.mm(input2, t3)
        t6 = torch.mm(input3, t3)
        t7 = torch.mm(input4, t3)
        return t7 - t4 + torch.mm(input4, t5) + torch.mm(input2, t6) - torch.mm(input1, t7)
# Inputs to the model
input1 = torch.randn(10, 10)
input2 = torch.randn(10, 10)
input3 = torch.randn(10, 10)
input4 = torch.randn(10, 10)
