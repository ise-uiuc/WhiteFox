
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 4
        self.head_dimension = 64
        self.query = torch.nn.Linear(self.head_dimension, self.head_dimension)
        self.key = torch.nn.Linear(self.head_dimension, self.head_dimension)
        self.value = torch.nn.Linear(self.head_dimension, self.head_dimension)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, q, k, v, scale_factor):
        q = self.query(q).reshape((q.shape[0], self.num_heads, -1))
        k = self.key(k).reshape((k.shape[0], self.num_heads, -1))
        v = self.value(v).reshape((v.shape[0], self.num_heads, -1))
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        qk = torch.matmul(q, k)
        scaled_qk = qk.div(scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output.reshape((q.shape[0], -1))

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 24, 128) # batch_size * length_q, head_dimension
x2 = torch.randn(1, 8, 128) # batch_size * length_kv, head_dimension
x3 = torch.randn(1, 8, 128) # batch_size * length_kv, head_dimension
