
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Dropout(0.2)
        self.b = torch.nn.Dropout(0.2)
        self.c = torch.nn.Dropout(0.2)
        self.d = torch.nn.Dropout(0.2)
        self.e = torch.nn.Dropout(0.2)
        self.f = torch.nn.Dropout(0.2)
        self.g = torch.nn.Dropout(0.2)
        self.h = torch.nn.Dropout(0.2)
        self.i = torch.nn.Dropout(0.2)
        self.j = torch.nn.Dropout(0.2)
        self.k = torch.nn.Dropout(0.2)
        self.l = torch.nn.Dropout(0.2)
        self.m = torch.nn.Dropout(0.2)
        self.n = torch.nn.Dropout(0.2)
        self.o = torch.nn.Dropout(0.2)
        self.p = torch.nn.Dropout(0.2)
        self.q = torch.nn.Dropout(0.2)
        self.r = torch.nn.Dropout(0.2)
        self.s = torch.nn.Dropout(0.2)
    def forward(self, x1):
        x3 = self.a(x1) + self.b(x1) + self.c(x1) + self.d(x1) + self.e(x1) + self.f(x1) + self.g(x1) + self.h(x1) + self.i(x1) + self.j(x1) + self.k(x1) + self.l(x1) + self.m(x1) + self.n(x1) + self.o(x1) + self.p(x1) + self.q(x1) + self.r(x1) + self.s(x1)
        x4 = torch.tensor([self.a(x1), self.b(x1), self.c(x1), self.d(x1), self.e(x1), self.f(x1), self.g(x1), self.h(x1), self.i(x1), self.j(x1), self.k(x1), self.l(x1), self.m(x1), self.n(x1), self.o(x1), self.p(x1), self.q(x1), self.r(x1), self.s(x1)])
        x5 = self.a(x1) + self.b(x1) + self.c(x1) + self.d(x1) + self.e(x1) + self.f(x1) + self.g(x1) + self.h(x1) + self.i(x1) + self.j(x1) + self.k(x1) + self.l(x1) + self.m(x1) + self.n(x1) + self.o(x1) + self.p(x1) + self.q(x1) + self.r(x1) + self.s(x1)
        x2 = torch.rand_like(x5)
        return (x3, x2, x4)
# Inputs to the model
x1 = torch.randn(2, 3, 3)
