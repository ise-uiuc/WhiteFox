
class Model(torch.nn.Module):
    def forward(self, input1):
        t1 = torch.mm(input1.transpose(0, 1), input1)
        t2 = torch.mm(input1, input1)

        t3 = torch.mm(input1, input1)
        t4 = torch.mm(input1.transpose(0, 1), input1)
        t5 = t3 + t4
        return t2 + t5
# Inputs to the model
input1 = torch.randn(6, 6)
