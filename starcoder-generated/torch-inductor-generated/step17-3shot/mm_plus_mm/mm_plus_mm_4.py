
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input1)
        t2 = torch.mm(input2, input2)
        t3 = torch.mm(input1, input1)
        t4 = torch.mm(input2, input2)
        t5 = t1 + t2
        t6 = t3 + t4
        return torch.cat((t5, t6), dim=1)
# Inputs to the model
input1 = torch.randn(9, 9, dtype=torch.float64)
input2 = torch.randn(9, 9, dtype=torch.float64)
