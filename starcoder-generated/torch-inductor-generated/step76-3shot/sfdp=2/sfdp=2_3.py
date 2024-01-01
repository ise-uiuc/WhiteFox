
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, val, inv_scale_factor=None, dropout_p=0.0):
        qk = torch.matmul(query, key.transpose(-2, -1))
        if inv_scale_factor is not None:
            scaled_qk = qk.div(inv_scale_factor)
        else:
            scaled_qk = qk
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, config.num_heads, 8, 8)
key = torch.randn(1, 8, config.num_heads, 16, 16)
val = torch.randn(1, 8, config.num_heads, 16, 16)
