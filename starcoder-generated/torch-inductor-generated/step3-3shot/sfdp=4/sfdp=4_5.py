
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(8, 8, bias=True)
        self.key = torch.nn.Linear(8, 8, bias=True)
        self.value = torch.nn.Linear(8, 8, bias=True)
 
    def forward(self, query, key, value, attn_mask=None):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        k = torch.transpose(k, -2, -1)
        qk = q @ k
        qk = qk / math.sqrt(q.size(-1))
        if(attn_mask!= None):
            qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
 
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(8, 10, 8)
key = torch.randn(8, 20, 8)
value = torch.randn(8, 20, 8)
