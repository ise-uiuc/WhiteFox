
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5):
        q2 = torch.matmul(x1, x2.transpose(-2, -1))
        q3 = torch.matmul(x3, x4.transpose(-2, -1))
        x4 = x5.transpose(-2, -1)
        qd4, qd3, qd2 = q3.div(x4), q2.div(x4), q2.div(x3)
        x5 = torch.matmul(qd5, x5)
        return qd3 

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 1)
x2 = torch.randn(1, 1, 1)
x3 = torch.randn(1, 1, 1)
x4 = torch.randn(1, 1, 1)
x5 = torch.randn(1, 1, 1)
__output1__, __output2__, __output3__, __output4__ = m(x1, x2, x3, x4, x5)

