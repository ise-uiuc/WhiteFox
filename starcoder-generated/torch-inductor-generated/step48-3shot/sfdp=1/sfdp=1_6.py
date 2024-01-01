
class Model(torch.nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert (d_model % nhead) == 0
        hidden_dims = d_model // nhead
        self.attention = torch.nn.MultiheadAttention(hidden_dims, nhead, dropout=dropout)
 
    def forward(self, query, key, value, scale_factor=1.):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = 1. / scale_factor
        scale_factor = inv_scale_factor
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m1 = Model(d_model, nhead, dropout)

# Inputs to the model
query = torch.randn(1, 1, d_model)
key = torch.randn(1, 1, d_model)
value = torch.randn(1, 1, d_model)
