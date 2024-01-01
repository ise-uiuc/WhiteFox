
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5, input6):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = t1 + t2
        t4 = torch.mm(input5, input6)
        t5 = t3 + t4
        return t5
# Inputs to the model
input1 = torch.randn(24, 24)
input2 = torch.randn(24, 24)
input3 = torch.randn(24, 24)
input4 = torch.randn(24, 24)
input5 = torch.randn(24, 24)
input6 = torch.randn(24, 24)
