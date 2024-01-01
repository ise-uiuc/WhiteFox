
class Model(torch.nn.Module):
    def __init__(
        self,
        query,
        key,
        value,
        scale_factor,
        dropout_p,
    ):
        super().__init__()
        num_heads = query.shape[0]
        emb_dim = query.shape[1]

        self.query = query
        self.key = key
        self.value = value
        self.dropout_p = dropout_p

        self.scale_factor = scale_factor
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.fc = torch.nn.Linear(num_heads * emb_dim, 1, bias=False)

    def forward(self):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = self.dropout(self.softmax(scaled_qk))
        dropout_qk = self.dropout(softmax_qk.matmul(self.value))

        return dropout_qk.matmul(self.fc(x.contiguous().view()))
         
# Initializing the model
query = torch.empty(2 * dim, dim)
key, value = (torch.empty(2 * dim, emb_dim), torch.empty(2 * dim, emb_dim))
scale_factor = torch.empty(2 * dim, 1)
dropout_p = torch.empty(1).uniform_()

m = Model(query, key, value, scale_factor, dropout_p)

# Initializing inputs
__x_len = 2
__y_len = 10
x1 = torch.empty(__x_len, __emb_dim)
x2 = torch.empty(__y_len, __emb_dim)

__scaled_qk__ = torch.matmul(x.reshape(1, -1), self.key.transpose(-2, -1).contiguous().view(__emb_dim, -1))
