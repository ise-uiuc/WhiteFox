 parameters
q = 4
d = 5
n = 6
batch_size = 5

# Model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Parameter(torch.randn(q, d))
        self.kv = torch.nn.Parameter(torch.randn(n, d))
        self.inv_scale = torch.nn.Parameter(torch.randn(d))
 
    def forward(self, x):
        query = self.q.unsqueeze(0).expand(batch_size, *self.q.size())
        kv = self.kv.unsqueeze(0).expand(batch_size, *self.kv.size())
        x1 = torch.matmul(query, kv.transpose(-2, -1))
        x2 = x1 / self.inv_scale
        x3 = x2.softmax(dim=-1)
        x4 = torch.matmul(x3, self.kv)
        return x4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(batch_size, q, d)
