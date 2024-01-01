


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2):
        t1 = input1
        t2 = input2
        for i in range(20):
            t3 = torch.mm(t1, t1)
            t3 = t3 + t2
            t4 = torch.mm(t3, t1)
            t5 = t3 + t4
            t6 = torch.mm(t3, t4)
            t3 = t5 + t6
            t7 = t1 + t3
            t8 = t7 + t2
            t9 = t4 + t8
            t1 = t5 + t9
        return t1
# Inputs to the model
input1 = torch.randn(128, 128)
input2 = torch.randn(128, 128)
