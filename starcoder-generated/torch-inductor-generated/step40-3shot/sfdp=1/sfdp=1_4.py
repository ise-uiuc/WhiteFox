
class Model(torch.nn.Module):
    def __init__(self,
                 dim_query, dim_key, dim_value, num_heads,
                 dropout_p=0.0):
        super().__init__()
        self.scale_factor = dim_key ** -0.5
        self.linear_q = torch.nn.Linear(dim_query, dim_key * num_heads, bias=False)
        self.linear_k = torch.nn.Linear(dim_key, dim_key * num_heads, bias=False)
        self.linear_v = torch.nn.Linear(dim_value, dim_value * num_heads, bias=False)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value):
        q, k, v = self.linear_q(query), self.linear_k(key), self.linear_v(value)
        q = q.view(-1, q.size(1), self.num_heads, k.size(-1))
        k = k.view(-1, self.num_heads, k.size(1), k.size(-1))
        v = v.view(-1, self.num_heads, v.size(1), v.size(-1))
 
        qk = torch.matmul(q, k.transpose(-2, -1)) # Compute the dot product of the query and key tensors
 
        scaled_qk = qk.div(self.scale_factor) # Scale the dot product by the scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
 
        dropout_qk = self.dropout(softmax_qk)
        return torch.matmul(dropout_qk, v)

# Initializing the model
model = Model(dim_query=2, dim_key=3, dim_value=4, num_heads=2)

# Inputs to the model
query = torch.randn(2, 2, 2)
key = torch.randn(2, 3, 3)
value = torch.randn(2, 3, 4)
