
class Model(torch.nn.Module):
    def forward(self, t1, t2, t3, t4):
        v1 = torch.mm(t1, t3)
        v2 = torch.mm(t3, t2)
        v3 = torch.mm(t1, t4)
        v4 = torch.mm(t4, t2)
        v5 = torch.mm(t3, t4)
        v6 = torch.mm(input1, t3)
        v7 = torch.mm(input2, t4)
        v8 = torch.mm(input1, t1)
        v9 = torch.mm(input2, t2)
        return t3
# Inputs to the model
input1 = torch.randn(33, 33)
input2 = torch.randn(33, 33)
t1 = torch.mm(input1, input1)
t2 = torch.mm(input1, input1)
t3 = torch.mm(input1, t1)
t4 = torch.mm(input2, t1)
