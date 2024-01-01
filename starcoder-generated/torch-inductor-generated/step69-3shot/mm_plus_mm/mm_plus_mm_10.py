
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = torch.mm(t1, t1)
        t4 = torch.mm(t2, t2)
        t5 = t3 + t4
        return t5
# Inputs to the model
input1 = torch.rand(2, 4)
input2 = torch.rand(2, 4)
input3 = torch.rand(2, 4)
input4 = torch.rand(2, 4)
