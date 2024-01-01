
class Model(torch.nn.Module):
    def __init__(self, dim, inverse_scale_factor=1.0, dropout_p=0.0):
        super().__init__()
        self.dim = dim
        self.dropout = dropout_p
        self.dropout_fn = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=-1)
        self.scale_factor = 1 / (dim ** 0.5) * inverse_scale_factor
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scale_qk = qk * self.scale_factor
        softmax_qk = self.softmax(scale_qk)
        dropout_qk = self.dropout_fn(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
dim = 10
dim_head = 2
heads = dim // dim_head
dropout_p = 0.1

inverse_scale_factor = (heads ** -0.5)
m = Model(dim, inverse_scale_factor, dropout_p)

# Inputs to the model
query = torch.randn(1, dim)
key = torch.randn(1, dim)
value = torch.randn(1, dim)
