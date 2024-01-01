
class Model(torch.nn.Module):
    def __init__(self,
                 dim_query,
                 dim_key,
                 dim_value):
        super().__init__()
        self.dim_query = dim_query
        self.dim_key = dim_key
        self.dim_value = dim_value
 
    def forward(self,
                query,
                key,
                value,
                dropout_p):
        inv_scale_factor = 1024 ** -0.25
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(dim_query=512,
          dim_key=64,
          dim_value=64)

# Inputs to the model
query = torch.randn(1, 8, 512)
key = torch.randn(1, 8, 64)
value = torch.randn(1, 8, 64)
dropout_p = 0.15
