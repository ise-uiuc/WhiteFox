
class Model(torch.nn.Module):
    def __init__(self,
                 query_dim, key_dim,
                 num_hidden_layers=2,
                 num_attention_heads=2):
        super().__init__()
        assert(num_hidden_layers > 0)
        assert(query_dim % num_attention_heads == 0)
        assert(key_dim % num_attention_heads == 0)
        self.num_attention_heads = num_attention_heads
        self.fcq = torch.nn.Linear(query_dim, query_dim, bias=False)
        self.fck = torch.nn.Linear(key_dim, query_dim, bias=False)
        self.fcv = torch.nn.Linear(key_dim, key_dim, bias=False)
        self.out = torch.nn.Linear(num_attention_heads * key_dim, key_dim, bias=False)

    def forward(self, query, key, value, inv_scale_factor=None, dropout_p=0.1):
        if dropout_p > 0:
            dropout_p /= self.num_attention_heads
        x1 = self.fcq(query)
        x2 = self.fck(key)
        x3 = self.fcv(value)
        x4 = torch.matmul(x1, x2.transpose(-2, -1))
        if inv_scale_factor is not None:
            x4 = x4.div(inv_scale_factor)
        x5 = x4.softmax(dim=-1)
        if dropout_p > 0:
            x6 = torch.nn.functional.dropout(x5, p=dropout_p)
        else:
            x6 = x5
        x7 = torch.matmul(x6, x3)
        x8 = self.out(x7.reshape(list(query.shape[:-1]) + [-1]))
        return x8

# Initializing the model
m = Model(query_dim=8, key_dim=16)

# Inputs to the model
query  = torch.randn(1, 8)
key    = torch.randn(2, 16)
value  = torch.randn(2, 16)
