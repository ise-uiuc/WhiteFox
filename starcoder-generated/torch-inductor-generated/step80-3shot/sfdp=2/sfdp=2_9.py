
class Model(torch.nn.Module):
    def __init__(self, query_len, key_len, value_len, heads, d_model, dropout_p=0.1):
        super().__init__()
        self.w_query = torch.nn.Parameter(torch.randn(query_len, heads, d_model) / np.sqrt(d_model), requires_grad=True)
        self.w_key = torch.nn.Parameter(torch.randn(d_model, key_len, heads) / np.sqrt(d_model), requires_grad=True)
        self.w_value = torch.nn.Parameter(torch.randn(d_model, value_len, heads) / np.sqrt(d_model), requires_grad=True)
        self.dropout_m = torch.dropout(0)
 
    def forward(self, query, key, value):
        dropout = self.dropout_m()
        q = bmm1d(dropout(self.w_query), query.transpose(-2, -1))
        k = bmm1d(dropout(self.w_key), key)
        v = bmm1d(dropout(self.w_value), value)
        scaled_qk = q.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = bmm1d(dropout_qk, v)
        return output

# Initializing the model
m = Model(10, 20, 30, 4, 512)

# Inputs to the model
query = torch.randn(8, 23, 512)
key = torch.randn(8, 4, 23, 512)
value = torch.randn(8, 4, 30, 512)
dropout_p = 0.3
