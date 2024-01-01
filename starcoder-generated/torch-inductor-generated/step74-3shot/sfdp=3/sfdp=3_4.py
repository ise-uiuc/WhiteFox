
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w0 = torch.nn.Linear(dim, dim)
        self.w1 = torch.nn.Linear(dim, dim)
        self.w2 = torch.nn.Linear(dim, dim)
        self.w3 = torch.nn.Linear(dim, dim)
 
    def forward(self, x0, x1, x2):
        v0 = self.w0(x0)
        v1 = self.w1(x1)
        v2 = self.w2(x2)
        v3 = F.gelu(v0 + v1, approximate=False)
        v4 = self.w3(v3 + v2)
        return v4

# Initializing the model
m = Model()

# Input tensors for the model
x0, x1, x2 = torch.randn(10, dim), torch.randn(10, dim), torch.randn(10, dim)
