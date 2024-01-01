
class Model(torch.nn.Module):
    def forward(self, input1, input2, input, input3, input4):
        t1 = torch.mm(input, input2)
        t2 = torch.mm(input3, input2)
        t3 = torch.mm(input1, input4)
        t4 = torch.mm(input4, input4)
        t5 = t1 + t2 + t3 + t4
        return t5
# Inputs to the model
input1 = torch.randn(8, 8)
input2 = torch.randn(8, 8)
input = torch.randn(8, 8)
input3 = torch.randn(8, 8)
input4 = torch.randn(8, 8)
