
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, attn_mask):
        qk = torch.matmul(query, key.permute(0,1,3,2))
        qk = qk / math.sqrt(64)
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = torch.matmul(attn_weight, value)
        return output, attn_weight

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn([1, 8, 64])
key = torch.randn([1, 8, 64])
value = torch.randn([1, 8, 64])
attn_mask = torch.triu(torch.ones([1, 64, 64]), diagonal=1) == 1
__output__, __attn_weight__ = m(query, key, value, attn_mask)

# Inputs to the model
query = torch.randn([1, 8, 64])
key = torch.randn([1, 8, 64])
value = torch.randn([1, 8, 64])
attn_mask = torch.randn([1, 1, 1]) > 0
__output__, __attn_weight__ = m(query, key, value, attn_mask)

