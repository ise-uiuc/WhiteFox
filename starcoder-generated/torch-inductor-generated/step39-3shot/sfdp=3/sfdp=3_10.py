
class Model(torch.nn.Module):
    def __init__(self, num_heads, key_dim, dropout_p):
        super(Model, self).__init__()
        
        self.embedding = np.float32(np.random.normal(loc=0.0, scale=0.01,size=(64, num_heads, key_dim)).astype(np.float32))
        self.query = torch.randn(1, 3, 64)
        self.key = None
        self.value = None
        self.scale_factor = 1. / np.sqrt(key_dim)
        self.dropout_p = dropout_p

    def forward(self):
        self.key = torch.tensor(self.embedding)
        self.value = torch.tensor(self.embedding)
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = value.matmul(dropout_qk)
        return output

# Initializing the model
m = Model(2, 3, 0.0)

# Inputs to the model

