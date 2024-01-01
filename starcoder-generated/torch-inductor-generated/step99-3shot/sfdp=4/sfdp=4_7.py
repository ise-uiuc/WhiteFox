
class Bai(torch.nn.Module):
    def __init__(self, N=1, M=2):
        super().__init__()
        self.f = torch.nn.Conv1d(N, M, 1, 1)
        self.g = torch.nn.Conv1d(N, M, 1, 1)
        self.h = torch.nn.Conv1d(N, M, 1, 1)
        self.v = torch.nn.Conv1d(M, N, 1, 1)
        self.r = torch.nn.Conv1d(M, N, 1, 1)
        self.t = torch.nn.Conv1d(M, N, 1, 1)
    def forward(self, x):
        a = self.f(x)
        b = self.g(x)
        c = self.h(x)
        # a, b, c are M x N x L => u is M x L x N
        u = self.v(a) + self.r(b) + self.t(c)
        return u
# Inputs to the model
x = torch.randn(1, 2304, 7, 7)
