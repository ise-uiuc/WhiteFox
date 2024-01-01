
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = AttentionLayer(hidden_dim, num_heads, dropout_p)
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
attn = AttentionLayer(hidden_dim, num_heads, dropout_p)

# Inputs to the model
q = torch.randn(batch_size, q_len, hidden_dim)
k = torch.randn(batch_size, k_len, hidden_dim)
v = torch.randn(batch_size, v_len, hidden_dim)
