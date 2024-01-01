
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5, input6):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = torch.mm(input5, input6)
        t4 = t1 + t2 + t3
        return t4
# Inputs to the model
input1 = torch.randn(22, 22)
input2 = torch.randn(22, 22)
input3 = torch.randn(22, 22)
input4 = torch.randn(22, 22)
input5 = torch.randn(22, 22)
input6 = torch.randn(22, 22)
