
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(4, 2)
        self.key = torch.nn.Linear(2, 3)
        self.value = torch.nn.Linear(3, 2)
 
    def forward(self, query, key, value, attn_mask):
        v1 = self.query(query)
        v2 = self.key(key)
        v3 = self.value(value)
        v4 = v1 @ v2.transpose(-2, -1) / math.sqrt(v1.size(-1))
        v5 = v4 + attn_mask
        attn_weight = F.softmax(v5, dim=v5.size(-1), dtype=torch.float32)
        attn_weight = F.dropout(attn_weight, p=0.1, training=True)
        v6 = attn_weight @ v3
        return v6

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 4)
key = torch.randn(2, 2)
value = torch.randn(3, 2)
attn_mask = torch.ones(2, 3)
m(query, key, value, attn_mask)
<torch.Tensor: shape=(1, 2, 2), dtype=float32, 
## Tensor Shape
- __Output__ 1: (1, 2, 2)
- __Output__ 2: (2, 3)
- __Output__ 3: (2, 4)
- __Output__ 4: (2)
