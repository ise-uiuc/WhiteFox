
class Model(torch.nn.Module):
    def __init__(self, p, q, r, s, u):
        super(Model, self).__init__()
        self.p = nn.Conv2d(p, r, kernel_size=2, stride=1)
        self.q = nn.Conv2d(q, s, kernel_size=1, stride=1)
        self.r = nn.Conv2d(r, u, kernel_size=3, stride=1)
        self.s = self.r.register_parameter('data', torch.randn(u, r, 3, 3).type(dtype))
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input1, input2)
        t3 = t2 + self.s(input2)
        return self.q(input2) + t1
# Inputs to the model
input1 = torch.randn(3, 3)
input2 = torch.randn(3, 3)
