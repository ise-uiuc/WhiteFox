
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input2, input1)
        t2 = t1 + torch.mm(input4, input1)
        t3 = torch.mm(input2, input4)
        t4 = torch.mm(input3, input1) + torch.mm(input3, input2)
        t5 = t3 + t4
        return t5
# Inputs to the model
input1 = torch.randn(4, 6)
input2 = torch.randn(4, 6)
input3 = torch.randn(6, 4)
input4 = torch.randn(4, 6)
