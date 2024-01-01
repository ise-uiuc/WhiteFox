
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask):
        attention_scaled_dot_product = torch.matmul(q, k.transpose(2, 3))/math.sqrt(q.size(-1))
        attention = attention_scaled_dot_product + mask
        attention_weights = torch.softmax(attention, dim=-1)
        attention_weights = torch.dropout(attention_weights, p=dropout_p, training=True)
        output = torch.matmul(attention_weights, v)
        return output

# Initializing the model
q = torch.randn(batch_size, n_heads, q_len, d_v)
k = torch.randn(batch_size, n_heads, d_k, k_dim)
v = torch.randn(batch_size, n_heads, v_len, d_v)
mask = torch.randn(batch_size, n_heads, q_len, k_len)
