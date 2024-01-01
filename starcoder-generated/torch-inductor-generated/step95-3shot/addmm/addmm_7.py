
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, inp, x2):
        t1 = torch.mm(x1, inp) # t1 is a 256x1 tensor result of matrix multiplication
        t2 = torch.mm(t1, x2) # t2 is a 256x1024 tensor result of another matrix multiplication
        return t2 # return t2
# Inputs to the model
x1 = torch.randn(256, 1)
inp = torch.randn(256, 256)
x2 = torch.randn(256, 1024)
