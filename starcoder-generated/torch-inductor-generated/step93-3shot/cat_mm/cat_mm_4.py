
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        t1 = torch.mm(input1, input2)
        t2 = torch.cat([t1, t1, t1, t1, t1], 1)
        t3 = torch.mm(input1, 2.0*input2)
        t4 = torch.cat([t3, t3, t3, t3, t3], 1)
        t5 = torch.mm(input1, 3.0*input2)
        t6 = torch.cat([t5, t5], 1)
        return torch.cat([t6, t4, t2], 1)


# Inputs to the model
input1 = torch.randn(10, 5)
input2 = torch.randn(5, 10)
