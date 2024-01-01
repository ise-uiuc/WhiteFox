
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input3)
        t3 = torch.mm(input1, input2)
        t4 = torch.mm(input3, input3)
        t5 = t1 + t2 + t3 + t4
        return t5
# Inputs to the model
input1 = torch.randn(16, 16)
input2 = torch.randn(16, 16)
input3 = torch.randn(16, 16)
input4 = torch.randn(16, 16)
input5 = torch.randn(16, 16)
