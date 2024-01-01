
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5, input6):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = torch.mm(input5, input6)
        t4 = torch.mm(input2, input2)
        return t1 + t2 + t3 + t4
# Inputs to the model
input1 = torch.randn(8, 16)
input2 = torch.randn(8, 8)
input3 = torch.randn(8, 8)
input4 = torch.randn(8, 8)
input5 = torch.randn(8, 8)
input6 = torch.randn(8, 8)
