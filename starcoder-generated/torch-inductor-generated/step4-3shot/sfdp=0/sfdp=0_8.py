
class Model(torch.nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.key = torch.nn.Linear(dim, dim)
        self.query = torch.nn.Linear(dim, dim)
 
    def forward(self, v1):
        k, q = self.key(v1), self.query(v1)
        inv_scale = math.sqrt(k.shape[-1])
        v2 = torch.matmul(q, k.transpose(-2, -1)) / inv_scale
        v3 = v2.softmax(dim=-1)
        v4 = torch.matmul(v3, v1)
        return v4
    
# Initializing the model
m = Model()

# Inputs to the model
v1 = torch.randn(2, 50, 128)
