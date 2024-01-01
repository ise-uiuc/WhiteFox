
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1):
        list_t1 = []
        for _ in range(5):
            t1 = torch.mm(input1, input1)
            list_t1.append(t1)
        t2 = torch.cat(list_t1, 0)
        list_t3 = []
        for _ in range(5):
            t3 = torch.mm(t2, t2)
            list_t3.append(t3)
        t4 = torch.cat(list_t3, 1)
        t5 = torch.mm(t4, t4)
        t6 = torch.cat([t5, t5, t5, t5], 1)
        return t6
# Inputs to the model
input1 = torch.randn(8, 8)
