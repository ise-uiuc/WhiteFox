
class Model(torch.nn.Module):
    def forward(self, q, k, v, attn_mask):
        v1 = torch.matmul(q, k.permute(0, 1, 3, 2))
        v2 = v1 / math.sqrt(query.size(-1))
        v3 = v2 + attn_mask
        v4 = torch.softmax(v3, dim=-1)
        v5 = torch.matmul(v4, v)
        return v5
 
# Initializing the model
m = Model()
 
# Inputs to the model
q = torch.randn(1, 1, 16)
k = torch.randn(1, 1, 64)
v = torch.randn(1, 1, 64)
attn_mask = torch.randint(0, 2, (1, 1, 16, 64)) # (1, 1, q_len, k_len)
