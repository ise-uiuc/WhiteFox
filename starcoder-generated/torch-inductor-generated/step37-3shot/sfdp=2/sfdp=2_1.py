
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Input to the model
query = torch.randn(q_len, bsz, attention_heads, 1, 256)
key = torch.randn(k_len, bsz, attention_heads, 256, 256)
value = torch.randn(k_len, bsz, attention_heads, 256, 256)
inv_scale_factor = torch.ones([bsz, attention_heads, 1, 1]).div(sqrt_dim)
