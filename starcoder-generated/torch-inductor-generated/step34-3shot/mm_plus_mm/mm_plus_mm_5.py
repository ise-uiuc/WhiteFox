
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = torch.mm(input1, input3)
        t4 = torch.mm(input2, input4)
        return t1 + t2 + t3 + t4
# Inputs to the model
input1 = torch.randn(3761992, 45)
input2 = torch.randn(154131, 3)
input3 = torch.randn(2375, 1)
input4 = torch.randn(2882, 10)
