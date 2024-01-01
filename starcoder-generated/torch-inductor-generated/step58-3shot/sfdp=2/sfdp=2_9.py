
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim=None):
        super().__init__()
        self.scale_factor = math.sqrt(query_dim)
 
    def forward(self, query, key, value, dropout_p, inv_scale_factor=1.0):
        if key is None:
            # key is None and value is None, this pattern will be dealt with later.
            inv_scale_factor = self.scale_factor
        else:
            if inv_scale_factor == 1.0:
                inv_scale_factor = self.scale_factor
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk / (inv_scale_factor ** 2)
        softmax_qk = scaled_qk.softmax(dim=-1)
        if dropout_p > 0:
            dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        elif dropout_p == 0:
            dropout_qk = softmax_qk
        else:
            raise ValueError("Invalid dropout probability.")
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(query_dim, key_dim)

# Inputs to the model
query = torch.randn(batch_size, num_heads, seq_length, query_dim)
key = torch.randn(batch_size, num_heads, seq_length, key_dim)
value = torch.randn(batch_size, num_heads, seq_length, query_dim)
dropout_p = 0.1
