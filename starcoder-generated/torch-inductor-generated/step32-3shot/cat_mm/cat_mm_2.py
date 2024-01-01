
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input2)
        t2 = torch.cat([t1, t1], 1)
        t3 = torch.cat([t2, t2], 1)
        t4 = torch.cat([t3, t3], 1)
        t5 = torch.cat([t4, t4], 1)
        t6 = torch.cat([t5, t5], 1)
        t7 = torch.cat([t6, t6], 1)
        t8 = torch.cat([t7, t7], 1)
        t9 = torch.cat([t8, t8], 1)
        return torch.cat([t9, t9], 1)
# Inputs to the model
input1 = torch.randn(32, 8)
input2 = torch.randn(8, 32)
