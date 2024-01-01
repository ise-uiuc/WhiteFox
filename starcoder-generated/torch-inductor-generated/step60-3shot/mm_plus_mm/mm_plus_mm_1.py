
class Model(torch.nn.Module):
    def forward(self, input1, input4, input3, input2):
        t1 = torch.mm(input1, input3)
        t2 = torch.mm(input1, input2)
        t3 = torch.mm(input4, input2)
        t4 = torch.mm(input4, input3)
        t5 = torch.mm(input3, input4)
        rpt1 = torch.mm(t1, torch.mm(t2, t3))
        rpt2 = torch.mm(t4, torch.mm(t2, t5))
        return rpt1 - rpt2
# Inputs to the model
input1 = torch.randn(3, 3, requires_grad=True)
input2 = torch.randn(3, 3, requires_grad=True)
input3 = torch.randn(3, 3, requires_grad=True)
input4 = torch.randn(3, 3, requires_grad=True)
