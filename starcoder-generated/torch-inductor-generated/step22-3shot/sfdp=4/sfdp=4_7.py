
class Model(torch.nn.Module):
    def forward(self, q, k, v, attn_mask):
        qk = torch.matmul(q, k.T) / math.sqrt(q.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = torch.matmul(attn_weight, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(8, 64, 80)
k = torch.randn(8, 24, 80)
v = torch.randn(8, 24, 80)
attn_mask = torch.zeros(8, 80, 80)
