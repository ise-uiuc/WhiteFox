
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query, key6, value3, mask, d_k):
        q = query # (batch_size, len_tgt, d_model)
        # get attention scores
        qk = torch.matmul(q, key6.transpose(-2, -1)) / math.sqrt(d_k)
        # (batch_size, len_q, len_k)
        qk = qk + mask
        # (batch_size, len_q, len_k)
        attn_weight = F.softmax(qk, dim=-1)
        # (batch_size, len_q, len_k)
        attn_vec = torch.matmul(attn_weight, value3)
        # (batch_size, len_q, d_model)
        # compute out projection
        out = self.linear_out(attn_vec) # (batch_size, len_q, d_model_tgt)
        return out
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
query1 = 1
key6 = 2
value3 = 2
# Model outputs
query = 1
key = 1
value = 1
mask = 1
