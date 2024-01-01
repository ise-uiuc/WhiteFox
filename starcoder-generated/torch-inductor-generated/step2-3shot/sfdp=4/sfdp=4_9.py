
class Model(torch.nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout):
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = n_head
        self.head_dim = d_k
 
        self.qk = torch.nn.Linear(d_model, self.embed_dim)
        self.v = torch.nn.Linear(d_model, self.embed_dim)
 
    def prepare_attentional_mechanism_inputs(self, query, key, value):
        qk = self.qk(query)
        v = self.v(value)
        return qk.view(qk.shape[0], self.num_heads, self.head_dim), v.view(v.shape[0], self.num_heads, self.head_dim)
 
    def dot_product_attention(self, q, k, v, attn_mask):
        qk = torch.matmul(q, k.transpose(-2, -1))
        qk = qk / math.sqrt(q.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk.transpose(1, 2), dim=-1)
        output = torch.matmul(attn_weight, v)
        return output, attn_weight
 
    def attention(self, query, key, value, attn_mask):
        q, k, v = self.prepare_attentional_mechanism_inputs(query, key, value)
        output, attn_weight = self.dot_product_attention(q, k, v, attn_mask)
        return output.transpose(1, 2), attn_weight
 
    def forward(self, q, k, v, attn_mask):
        output, attn_weight = self.attention(q, k.transpose(-2, -1), v, attn_mask)
        output = output.reshape(output.shape[0], output.shape[1], self.embed_dim)
        return output, attn_weight

# Initializing the model
m = Model(32, 64, 16, 16, 0.5)

# Inputs to the model
q = torch.randn(4, 8, 64)
k = torch.randn(4, 2, 128)
v = torch.randn(4, 2, 256)
__output__, __attn_weight__ = m(q, k, v)

