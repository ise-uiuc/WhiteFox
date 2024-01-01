
class Model(torch.nn.Module):
    def __init__(self, in_dim, heads_num, mlp_dim):
        super().__init__()
        self.q = torch.nn.Linear(in_dim, in_dim)
        self.k = torch.nn.Linear(in_dim, in_dim)
        self.v = torch.nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout()
 
    def forward(self, query, key, value, scale_factor, dropout_p, mask):
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        #
        # You may assume that key_dim is equal to value_dim
        #
        qk = torch.matmul(q, k.t()) / scale_factor
        scaled_qk = qk.softmax()
        dropout_qk = self.dropout(scaled_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(64, 1, 128)

# Inputs to the model
query = torch.randn(1, 8, 64)
key = torch.randn(1, 8, 64)
value = torch.randn(1, 8, 64)
scale_factor = math.sqrt(8)
dropout_p = 0.25
mask = torch.randn(1, 1, 8)
