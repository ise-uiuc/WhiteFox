
class Model(torch.nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super().__init__()
        self.query_linear = torch.nn.Linear(dim_q, dim_k)
        self.key_linear = torch.nn.Linear(dim_k, dim_k)
        self.value_linear = torch.nn.Linear(dim_v, dim_v)
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        q = self.query_linear(query)
        k = self.key_linear(key)
        v = self.value_linear(value)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
dim_q = 16
dim_k = 18
dim_v = 20
m = Model(dim_q, dim_k, dim_v)

# Inputs to the model
query = torch.randn(8, 3, dim_q)
key = torch.randn(8, 3, dim_k)
value = torch.randn(8, 3, dim_v)
inv_scale_factor = torch.randn(1)
dropout_p = torch.randn(1)
