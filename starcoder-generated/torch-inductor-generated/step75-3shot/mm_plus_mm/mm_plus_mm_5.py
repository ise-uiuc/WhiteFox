
class Model(torch.nn.Module):
    def forward(self, input0, input1, input2):
        t1 = torch.mm(7*torch.ones_like(input0), input0)
        t2 = torch.matmul(7*torch.ones_like(input1), input1)
        t3 = torch.mm(input2, input0)
        t4 = torch.mm(t1, t3)
        return torch.mm(t2, t4)
# Inputs to the model
input0 = torch.rand(5, 5)
input1 = torch.rand(5, 5)
input2 = torch.rand(5, 5)
input3 = torch.rand(5, 5)
