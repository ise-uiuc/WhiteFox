
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.rand_like(x)
        t3 = torch.rand_like(t1, dtype=torch.int)
        t4 = torch.rand_like(t1, dtype=torch.bool)
        t5 = torch.rand_like(t1, dtype=torch.long)
        t6 = torch.rand_like(t1, dtype=torch.bfloat16)
        t7 = torch.rand_like(t1, dtype=torch.QInt8)
        t8 = torch.rand_like(t1, dtype=torch.quint8)
        t9 = torch.rand_like(t1, dtype=torch.qint32)

        return t1 + t3 + t4 + t5 + t6 + t7 + t8 + t9
# Input to the model
x = torch.randn(1, 2, 2)
