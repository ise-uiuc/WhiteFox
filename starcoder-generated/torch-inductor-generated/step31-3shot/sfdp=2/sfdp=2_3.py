
class Model(torch.nn.Module):
    def __init__(self, dim_query, dim_key, dim_value, dropout_p, inv_scale_factor):
        super().__init__()
        
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(dim_query=8192, dim_key=8192, dim_value=8192, dropout_p=0.1, inv_scale_factor=48.2)

# Inputs to the model
query = torch.randn(1, 8192, 32)
key = torch.randn(1, 8192, 32)
value = torch.randn(1, 8192, 32)
