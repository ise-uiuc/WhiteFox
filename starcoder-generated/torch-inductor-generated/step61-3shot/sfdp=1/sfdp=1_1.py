
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.inv_scale_factor = np.sqrt(key_dim)
        
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.matmul_1 = torch.nn.Linear(query_dim, key_dim, bias=False)
        self.matmul_2 = torch.nn.Linear(key_dim, value_dim, bias=False)

    def forward(self, q, k, v):
        q = q.unsqueeze(dim=1)
        q = self.matmul_1(q)
        k = self.matmul_1(k)
        qk = torch.matmul(q, k.transpose(-2, -1))
        qk = qk.div(self.inv_scale_factor)
        
        softmax_qk = self.softmax(qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(64, 64, 64)

# Inputs to the model
__query__ = torch.randn(1, 5, 64)
__key__ = torch.randn(5, 15, 64)
__value__ = torch.randn(5, 15, 64)
