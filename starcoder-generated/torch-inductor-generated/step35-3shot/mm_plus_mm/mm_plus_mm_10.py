
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input2, input1)
        t2 = torch.mm(input2, input1)
        t3 = torch.mm(input2, input1)
        t4 = torch.mm(input2, input1)
        t5 = torch.mm(input2, input1)
        return t1 + t2 + t3 + t4 + t5
# Inputs to the model
input1 = torch.randn(3, 3, requires_grad=True)
input2 = torch.randn(3, 3, requires_grad=True)
input3 = torch.randn(3, 3, requires_grad=True)
input4 = torch.randn(3, 3, requires_grad=True)
