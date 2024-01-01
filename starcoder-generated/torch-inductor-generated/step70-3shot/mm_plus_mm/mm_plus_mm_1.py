
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input1)
        t2 = torch.mm(input3, input3)
        t3 = torch.mm(input1, input3)
        t4 = torch.mm(input4, input3)
        t5 = torch.mm(input3, input3)
        return t1 + t2 
# Inputs to the model
input1 = torch.randn(2, 2)
input2 = torch.randn(2, 2)
input3 = torch.randn(2, 2)
input4 = torch.randn(2, 2)
