
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5, input6):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = t1 + t2
        t4 = t3 + input5
        t5 = t4 + input6
        return t5
# Inputs to the model
input1 = torch.randn(20, 20)
input2 = torch.randn(20, 20)
input3 = torch.randn(20, 20)
input4 = torch.randn(20, 20)
input5 = torch.randn(20, 20)
input6 = torch.randn(20, 20)
