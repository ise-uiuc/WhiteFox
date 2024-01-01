
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t1 = torch.mm(input1, input1)
        t2 = torch.mm(input2, input2)
        t3 = torch.mm(input3, input3)
        t4 = t2 + t3 + t1
        return t4
# Inputs to the model
input1 = torch.randn(2, 2)
input2 = torch.randn(2, 3)
input3 = torch.randn(3, 4)
