
class Model(torch.nn.Module):
    def __init__(self, dim, nHeads, num_qkv):
        super(Model, self).__init__()
        self.proj_q = torch.nn.Linear(dim, num_qkv * nHeads)
        self.proj_k = torch.nn.Linear(dim, num_qkv * nHeads)
        self.proj_v = torch.nn.Linear(dim, num_qkv * nHeads)
        self.proj_o = torch.nn.Linear(num_qkv * nHeads, dim)
 
    def forward(self, x1, x2, x3, x4):
        q = self.proj_q(x1)
        k = self.proj_k(x2)
        v = self.proj_v(x3)
        o = self.proj_o(x4)
        return o

# Initializing the model
m = Model(1024, 10, 10)

# Inputs to the model
x1 = torch.randn(4, 1024)
x2 = torch.randn(4, 1024)
x3 = torch.randn(4, 1024)
x4 = torch.randn(4, 10240)
