
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5, x6, x7):
        __t1__ = torch.matmul(x1, x3.transpose(-2, -1))
        __t2__ = torch.matmul(x2, x4.transpose(-2, -1))
        __t3__ = torch.matmul(x5, x6.transpose(-2, -1))
        __t4__ = __t1__ + __t2__
        __t5__ = __t3__.softmax(dim=-1)
        __t6__ = torch.matmul(__t5__, x7)
        return __t6__

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 28, 50)
x2 = torch.randn(1, 28, 50)
x3 = torch.randn(1, 50, 28)
x4 = torch.randn(1, 50, 28)
x5 = torch.randn(1, 198, 107)
x6 = torch.randn(1, 107, 198)
x7 = torch.randn(1, 2048, 2048)
