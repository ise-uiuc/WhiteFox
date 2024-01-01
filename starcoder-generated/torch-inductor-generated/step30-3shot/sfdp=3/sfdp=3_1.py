
class Model(torch.nn.Module):
    def __init__(self, n_heads, value_size, dropout_p):
        super().__init__()
        self.n_heads = n_heads
        self.value_size = value_size
        self.dropout_p = dropout_p
        self.softmax_d = Softmax(dim=1) # Specify the dimension of the softmax operation
        self.dropout = Dropout(dropout_p) # Specify the dropout probability

    def forward(self, query, key, value, scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk * scale_factor
        softmax_qk = self.softmax_d(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        qo = dropout_qk.matmul(value)
        return qo

from torch.nn.parameter import Parameter

class Model(torch.nn.Module):
    def __init__(self, n_heads, value_size, dropout_p):
        super().__init__()
        self.n_heads = n_heads
        self.value_size = value_size
        self.dropout_p = dropout_p
        self.weight = Parameter(torch.Tensor(1, n_heads, value_size, value_size)) # Specify the parameter shape
        self.softmax_q = Softmax(dim=1) # Specify the dimension of the softmax operation
        self.softmax_d = Softmax(dim=3) # Specify the dimension of the softmax operation
        self.dropout = Dropout(dropout_p) # Specify the dropout probability
        self.softmax_v = Softmax(dim=2) # Specify the dimension of the softmax operation

    def forward(self, query, key, value, scale_factor):
        wq = torch.matmul(query, self.weight)
        qk = torch.matmul(wq, key.transpose(-2, -1))
        scaled_qk = qk * scale_factor
        softmax_qk = self.softmax_q(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        dv = dropout_qk.matmul(value)
        softmax_dv = self.softmax_d(dv)
        dropout_dv = self.dropout(softmax_dv)
        softmax_dropout_dv = self.softmax_v(dropout_dv)
        output = softmax_dropout_dv.mul(dv)
        return output

# Initializing the model
m = Model(config.num_heads, config.hidden_dims, config.dropout_p)

# Inputs to the model
query = torch.randn(1, 100, 50)
key = torch.randn(1, 100, 40)
value = torch.randn(1, 100, 40)
m.forward(query, key, value, 1.0).size()

