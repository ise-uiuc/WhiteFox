
class Model(torch.nn.Module):
    def __init__(self, dropout_p, num_heads, d_key):
        super().__init__()
        self.dropout_p = dropout_p
        self.num_heads = num_heads
        self.d_key = d_key
        self.k = torch.nn.Linear(d_key, d_key, bias=False)
        self.q = torch.nn.Linear(d_key, d_key, bias=False)
        self.v = torch.nn.Linear(d_key, d_key, bias=False)
        self.softmax_d = -1
    
    def forward(self, x1):
        _shape = list(x1.shape)
        _shape[1] = self.d_key
        _shape = (_shape[1], _shape[0], _shape[2], _shape[3] // self.num_heads)
        key = self.k(x1).view(*_shape).transpose(0, 1)
        _shape[0] = self.d_key
        _shape = (_shape[1], _shape[0], _shape[2], _shape[3] // self.num_heads)
        query = self.q(x1).view(*_shape).transpose(0, 1)
        _shape[0] = self.d_key
        _shape = (_shape[1], _shape[0], _shape[2], _shape[3] // self.num_heads)
        value = self.v(x1).view(*_shape).transpose(0, 1)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(np.sqrt(qk.shape[-1])).softmax(dim=self.softmax_d)
        dropout_qk = torch.nn.functional.dropout(scaled_qk, p=self.dropout_p)
        _shape = list(value.shape)
        _shape[0] = value.shape[0] * value.shape[1]
        return dropout_qk.matmul(value.view(*_shape))

# Defining parameters
d_key = 1024
num_heads = 64
dropout_p = 0.1
 
# Initializing the model
m = Model(dropout_p, num_heads, d_key)

# Inputs to the model
x1 = torch.randn(1, d_key, 64, 64)
