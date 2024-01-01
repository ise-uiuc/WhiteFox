
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(inp_dim, inp_dim)
        self.scale = np.power(inp_dim, -0.5)
 
    def forward(self, x1, x2):
        q = 2 * x1 - 1
        k = self.proj(2 * x2 - 1)
        v = 2 * x2 - 1
        scaled_qkv = torch.matmul(q, k.transpose(-2, -1)).mul_(self.scale).softmax(dim=-1)
        output = scaled_qkv.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, inp_dim)
x2 = torch.randn(1, inp_dim)
