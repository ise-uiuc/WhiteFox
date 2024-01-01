
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
batch, heads, seq_len_q, seq_len_k, dim = 4, 2, 8, 8, 16
query = torch.randn(batch, heads, seq_len_q, seq_len_k, dim)
key = torch.randn(batch, heads, seq_len_k, seq_len_q, dim)
value = torch.randn(batch, heads, seq_len_k, seq_len_q, dim)
inv_scale_factor = torch.randn(1,1)
dropout_p = 0.5
