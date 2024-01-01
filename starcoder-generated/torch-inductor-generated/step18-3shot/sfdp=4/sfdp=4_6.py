
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, qk, attn_mask, value):
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output

# Initializing the model
m = Model()

# Generate a query of the size (batch_size, n_heads, seq_len_q, dim_qk)
qk = torch.randn(1, 2, 2, 16)

# Generate an attention mask of the size (batch_size, n_heads, seq_len_q, seq_len_q)
attn_mask = torch.randn(1, 2, 2, 2)

# Generate a value of the size (batch_size, n_heads, seq_len_v, dim_qk)
value = torch.randn(1, 2, 2, 16)
