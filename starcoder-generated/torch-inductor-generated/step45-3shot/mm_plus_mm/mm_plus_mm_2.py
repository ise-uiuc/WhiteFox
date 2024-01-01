
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5, input6, input7, input8):
        t1 = torch.mm(input4, torch.mm(input3,input2))
        t2 = t1 + torch.mm(input2, torch.mm(input3, input1))
        t3 = torch.mm(input6, torch.mm(input5, input4))
        t4 = t2 + t3
        t5 = torch.mm(input7, torch.mm(input6, input5))
        t6 =  torch.mm(input8, torch.mm(input7, input3))
        return t5 + torch.mm(input1, t6) + t4
# Inputs to the model
input1 = torch.randn(4, 4)
input2 = torch.randn(4, 4)
input3 = torch.randn(4, 4)
input4 = torch.randn(4, 4)
input5 = torch.randn(4, 4)
input6 = torch.randn(4, 4)
input7 = torch.randn(4, 4)
input8 = torch.randn(4, 4)
