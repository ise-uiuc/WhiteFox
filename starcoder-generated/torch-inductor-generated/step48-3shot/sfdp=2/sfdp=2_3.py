
class Model(torch.nn.Module):
    def __init__(self, attn_dim, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(8, 0.2)

# Inputs to the model
query = torch.randn(1, 1, 8)
key = torch.randn(1, 1, 8)
value = torch.randn(1, 1, 8)
inv_scale_factor = 1.0 / math.sqrt(8)
output = m(query, key, value, inv_scale_factor)

