
class Model(torch.nn.Module):
    def __init__(self, query_size, key_size, value_size, input_size, n_heads, dropout_p,
                 scale_factor):
        super().__init__()
 
        self.q = torch.nn.Linear(query_size, input_size*n_heads)
        self.k = torch.nn.Linear(key_size, input_size*n_heads)
        self.v = torch.nn.Linear(value_size, input_size*n_heads)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value):
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
 
        q = q.reshape(q.size(0), -1, q.size(3)).transpose(1, 2)
        k = k.reshape(k.size(0), -1, k.size(3)).transpose(1, 2)
        v = v.reshape(v.size(0), -1, v.size(3)).transpose(1, 2)
 
        q = q * (k.size(-1) ** -0.5)
 
        scaled_qk = torch.matmul(q, k.transpose(-2, -1))
        softmax_qk = scaled_qk.softmax(-1)
        dropout_qk = self.dropout(softmax_qk)
 
        output = torch.matmul(dropout_qk, v)
 
        output = output.transpose(1, 2).contiguous(). \
            view(output.size(0), -1, n_heads * output.size(3))
        return output

# Initializing the model
m = Model(100, 100, 100, 1024, 4, 0.05, 0.0625)
 
# Inputs to the model
query = torch.randn(256, 100)
key = torch.randn(256, 150)
value = torch.randn(256, 150)

