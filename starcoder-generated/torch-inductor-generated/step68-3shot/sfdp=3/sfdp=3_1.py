
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim=8, num_heads=2)
 
    def forward(self, x1, x2):
        q = self.attention.q_proj(x1)
        k = self.attention.k_proj(x2)
        v = self.attention.v_proj(x2)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = 1 / math.sqrt(q.size(-1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_p = 0.2
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
x2 = torch.randn(1, 8, 64, 64)
