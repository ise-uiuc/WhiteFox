
class Model(torch.nn.Module):
    def forward(self, inputs1, input2, input3, input4, input5):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = torch.mm(input1, input4)
        t4 = torch.mm(input3, input2)
        t5 = torch.mm(input5, input3)
        t6 = torch.mm(input5, input4)
        t7 = torch.mm(input5, input2)
        return t1 + t2 + t3 + t4 + t5 + t6 + t7
# Inputs to the model
inputs1 = torch.randn(2, 2)
input2 = torch.randn(2, 2)
input3 = torch.randn(2, 2)
input4 = torch.randn(2, 2)
input5 = torch.randn(2, 2)
