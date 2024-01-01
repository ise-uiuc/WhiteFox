
class Model(nn.Module):
    def forward(self, input1, input2, input3, input4, input5):
        t1 = torch.mm(input1, input4)
        t2 = torch.mm(input2, input3)
        t3 = torch.mm(input5, input3)
        t4 = torch.mm(input1, input4)
        t5 = torch.mm(input2, input3)
        t6 = torch.mm(input5, input3)
        t7 = t6 + t1
        t8 = torch.mm(t6, input3)
        return t7 + t2 + t3 + t4 + t5 + t8
# Inputs to the model
input1 = torch.randn(32, 4, 64)
input2 = torch.randn(32, 4, 64)
input3 = torch.randn(32, 4, 64)
input4 = torch.randn(32, 4, 64)
input5 = torch.randn(32, 4, 64)
