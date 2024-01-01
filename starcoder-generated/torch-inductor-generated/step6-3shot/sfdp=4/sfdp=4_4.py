
class Model(torch.nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, seq_len, num_encoder_layers, num_decoder_layers, batch_size):
        super().__init__()
        self.q_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.attn_weight_dropout = nn.Dropout(dropout)
 
    def forward(self, queries, keys, values, mask):
        queries = queries.reshape((batch_size, -1, queries.size(-1)))
        keys = keys.reshape((batch_size, -1, keys.size(-2), keys.size(-1)))
        values = values.reshape((batch_size, -1, values.size(-2), values.size(-1)))
        q = self.q_linear(queries)
        q = F.dropout(q, p=dropout, training=self.training)
        k = self.k_linear(keys)
        k = F.dropout(k, p=dropout, training=self.training)
        v = self.v_linear(values)
        v = F.dropout(v, p=dropout, training=self.training)
        weight = torch.matmul(q, k.transpose(2, 3))
        weight = weight / math.sqrt(hidden_dim)
        if mask is not None:
            mask = mask.reshape((batch_size, 1, seq_len, seq_len))
            weight = torch.where(mask > 0, weight, weight.new([float('-inf')]))
        weight = F.softmax(weight, dim=-1)
        weight = self.attn_weight_dropout(weight)
        output = torch.matmul(weight, v).reshape(batch_size, -1, hidden_dim)
        return ouput

# Initializing the model
m = Model(6, 128, 0.1, 100, 1, 1, 10)

# Inputs to the model
queries = torch.randn(10, 100, 128)
keys = torch.randn(10, 400, 128)
values = torch.randn(10, 400, 128)
mask = torch.randint(0, 2, (10, 1, 100, 400))
