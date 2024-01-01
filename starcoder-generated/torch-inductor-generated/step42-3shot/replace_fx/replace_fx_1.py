
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 10)
        self.linear2 = torch.nn.Linear(10, 364)
        self.linear3 = torch.nn.Linear(364, 10)
        self.linear4 = torch.nn.Linear(10, 2)
    def forward(self, x):
        o1 = self.linear1(x)
        o2 = torch.rand_like(o1, dtype=o1.dtype)
        o3 = self.linear2(torch.gelu(o1))
        o4 = torch.rand_like(o3, dtype=o1.dtype)
        o5 = self.linear3(o3)
        o6 = torch.rand_like(o5, dtype=o1.dtype)
        o7 = self.linear4(o5)
        o8 = torch.rand_like(o7, dtype=o1.dtype)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
