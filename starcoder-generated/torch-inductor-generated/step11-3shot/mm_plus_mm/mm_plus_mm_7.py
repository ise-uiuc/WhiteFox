
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5):
        t1 = torch.mm(input1, input4)
        t2 = torch.mm(input2, input3)
        t3 = t1 + t2
        t4 = torch.mm(input1, input2)
        t5 = torch.mm(input3, input4)
        t6 = torch.mm(input4, input5)
        t7 = t4 + t5 + t6 + t3
        return t7
# Inputs to the model
input1 = torch.randn(3, 3, dtype=torch.int32)
input2 = torch.randn(3, 3, dtype=torch.int32)
input3 = torch.randn(3, 3, dtype=torch.int32)
input4 = torch.randn(3, 3, dtype=torch.int32)
input5 = torch.randn(3, 3, dtype=torch.int32)
