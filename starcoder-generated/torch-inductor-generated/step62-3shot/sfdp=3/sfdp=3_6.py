
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_key_func = torch.nn.Linear(embed_dim, embed_dim)
        self.value_func = torch.nn.Linear(embed_dim, embed_dim)
        self.softmax_func = torch.nn.Softmax(dim=-1)
        self.dropout_func = torch.nn.Dropout(dropout)
 
    def forward(self, query, key, value):
        qk = self.query_key_func(query).matmul(key.transpose(-2, -1))
        scale_factor = self.query_key_func.embedding_dim ** (-0.5)
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = self.softmax_func(scaled_qk)
        dropout_qk = self.dropout_func(softmax_qk)
        ret = dropout_qk.matmul(value)
        return ret

# Initializing the model
m = Model()
embed_dim = 4
dropout = 0.
query = torch.randn(10, 3, embed_dim)
key = torch.randn(10, 3, embed_dim)
value = torch.randn(10, 3, embed_dim)
