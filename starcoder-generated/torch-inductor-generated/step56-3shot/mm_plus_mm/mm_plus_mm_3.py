
class Model(torch.nn.Module):
     def forward(self, input1, input2):
        t1 = torch.mm(input1, input1)
        t2 = torch.mm(input2, input2)
        t2 = torch.mm(t2, t1)
        t3 = torch.mm(input1, input2)
        t1 = t1 + input1
        t2 = t2 + input2
        t3 = t3 + t2
        return t1 + t2 + t3
# Inputs to the model
input1 = torch.randn(298, 298)
input2 = torch.randn(298, 298)
