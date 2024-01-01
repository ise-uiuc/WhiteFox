
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, n_head):
        super(Model, self).__init__()
        self.query = torch.nn.Parameter(torch.rand(n_head, query_dim))
        self.key = torch.nn.Parameter(torch.rand(n_head, key_dim))
        self.dropout = torch.nn.Dropout(dropout_rate)
    
    def forward(self, query, value):
        scale_factor = 1. / (query.shape[-1] ** 0.5)
        qk = torch.matmul(query, self.key.transpose(-2,-1)) 
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
n_head = 16
query_dim = 32
key_dim = 32
m = Model(query_dim, key_dim, n_head)

# Inputs to the model
query = torch.randn(batch_size, n_head, query_dim)
value = torch.randn(seq_len, query_dim)
