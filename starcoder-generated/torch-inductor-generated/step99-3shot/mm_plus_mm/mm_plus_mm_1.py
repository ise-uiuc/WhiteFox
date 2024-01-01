
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5, input6):
        t0 = torch.mm(input4, input6)
        t1 = torch.mm(input3, input2)
        t2 = torch.mm(input5, input4)
        t0 = torch.mm(input1, input5)
        t3 = torch.mm(input2, input1)
        t4 = t1 + t2 + t0 + t3 + t0 + t2
        return t4
# Inputs to the model
input1 = torch.randn(5, 5)
input2 = torch.randn(5, 5)
input3 = torch.randn(5, 5)
input4 = torch.randn(5, 5)
input5 = torch.randn(5, 5)
input6 = torch.randn(5, 5)
