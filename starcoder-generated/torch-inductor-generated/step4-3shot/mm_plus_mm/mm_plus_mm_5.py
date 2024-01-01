
class Model(torch.nn.Module):
    def __int__(self):
        super().__init__()
    def forwad(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = t1.mm(t2)
        return t3
# Inputs to the model
input1 = torch.randn(1, 1)
input2 = torch.randn(1, 1)
input3 = torch.randn(1, 1)
input4 = torch.randn(1, 1)
