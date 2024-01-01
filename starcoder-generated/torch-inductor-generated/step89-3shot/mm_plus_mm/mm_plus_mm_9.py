
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5):
        t1 = torch.mm(input1, input5)
        t2 = torch.mm(input2, input3)
        t3 = input2 + input4
        t4 = input1 + input3
        t5 = torch.mm(t1, t3.mm(input2))
        t6 = torch.mm(t1, t4.mm(input3))
        return t5 + t6
# Inputs to the model
input1 = torch.randn(8, 8)
input2 = torch.randn(16, 32)
input3 = torch.randn(32, 16)
input4 = torch.randn(64, 32)
input5 = torch.randn(16, 8)
