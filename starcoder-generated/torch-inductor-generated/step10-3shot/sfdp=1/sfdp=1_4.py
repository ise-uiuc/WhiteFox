
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_nn = torch.nn.Dropout(dropout_p)

    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout_nn(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 1, embed_dim)
key = torch.randn(1, num_heads, seq_len, embed_dim)
value = torch.randn(1, num_heads, seq_len, embed_dim)
inv_scale_factor = torch.randn(1, num_heads, 1, 1)
