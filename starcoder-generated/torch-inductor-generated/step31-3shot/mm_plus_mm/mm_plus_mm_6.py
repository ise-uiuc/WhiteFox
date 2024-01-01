
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5, input6):
        t0 = torch.mm(input2, input1)
        t1 = torch.mm(input4, input3)
        t2 = torch.mm(input4, input5)
        t3 = torch.mm(input4, input6)
        t4 = torch.mm(t3, t1)
        t5 = t0 + t2
        t6 = t0 * t5
        t7 = t4 * t6
        return t7
# Inputs to the model
input1 = torch.randn(5, 5)
input2 = torch.randn(5, 5)
input3 = torch.randn(5, 5)
input4 = torch.randn(5, 5)
input5 = torch.randn(5, 5)
input6 = torch.randn(5, 5)
