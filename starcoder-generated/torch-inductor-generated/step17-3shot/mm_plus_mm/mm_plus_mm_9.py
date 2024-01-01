
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        t1 = input1
        t2 = input2
        t3 = t1
        for i in range(100):
            if i % 2 == 0:
                t3 = t1 + torch.mm(t2, t3)
            else:
                t3 = t1 + torch.mm(t3, t2)
        return t3 + t1
# Inputs to the model
input1 = torch.randn(4, 10)
input2 = torch.randn(10, 4)
