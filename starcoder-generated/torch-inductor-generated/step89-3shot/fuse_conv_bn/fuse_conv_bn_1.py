
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(1, 1, 1)
        self.b = nn.Conv2d(3, 3, 3)
        self.f = MyFunc.apply
        self.c = nn.Conv2d(1, 1, 1)
    def forward(self, x):
        y = F.relu(self.a(x))
        y = self.b(y)
        return self.f(y)
    @staticmethod
    def symbolic(graph, *args):
        return torch._C._get_symbolic_trace_graph("aten::relu", graph, args)
    @staticmethod
    def forward(x):
        return x >= 0
# Inputs to the model
x = torch.randn(3, 1, 2, 3)
