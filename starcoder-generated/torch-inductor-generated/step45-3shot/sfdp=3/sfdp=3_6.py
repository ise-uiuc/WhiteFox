
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(768, 384)
        self.layernorm1 = nn.LayerNorm(384, eps=1E-12)
        self.layernorm2 = nn.LayerNorm(768, eps=1E-12)
        self.linear1 = nn.Linear(384, 768)
        self.linear2 = nn.Linear(768, 3072)
        self.dropout = nn.Dropout(0.1)
 
    def forward(self, query, key, vl):
        q = self.embed(query)
        k = self.embed(key)
        v = self.embed(vl)
        
        q = self.layernorm1(q)
        k = self.layernorm1(k)
        v = self.layernorm1(v)
        
        qkv = torch.einsum('bnij,bmij->bnmi', q, k)
        qkv = self.linear2(qkv)

        qkv += q.unsqueeze(2) + k.unsqueeze(1)
        qkv = qkv * float(np.sqrt(1. / 385))
        qkv = torch.matmul(qkv, v.transpose(2, 1))

        q = q.matmul(qkv.transpose(2, 3))
        k = torch.matmul(qkv, k.transpose(2, 1))
        v = torch.matmul(qkv, v.transpose(2, 1))
        qkv = torch.cat([q, k, v], dim=2)

        qkv = qkv + self.linear1(qkv)
        return self.dropout(qkv)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 384, 768) # Query
x2 = torch.randn(5, 384, 768) # Key
x3 = torch.randn(5, 384, 768) # Value
