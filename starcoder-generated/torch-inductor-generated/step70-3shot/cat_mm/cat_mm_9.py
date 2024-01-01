
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2, input3):
        t1 = torch.mm(input1, input2)
        t2 = torch.cat([t1, t1], 1)
        t3 = torch.cat([t2, t2], 3)
        t4 = torch.cat([t2, t2, t2], 1)
        t5 = torch.cat([t2, t2], 1)

        t6 = torch.cat([t3, t4, t5], 0)
        t7 = torch.cat([t2, t6], 2)
        t8 = torch.cat([t7, t7, t7, t7], 1)
        return torch.cat([t5, t8, t8, t8], 1)
# Inputs to the model
input1 = torch.randn(32, 1)
input2 = torch.randn(1, 32)
input3 = torch.randn(32, 32)
