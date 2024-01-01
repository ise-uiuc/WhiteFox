
class Model(torch.nn.Module):
    def __init__(self, q_shape, k_shape, v_shape, dropout=0):
        super().__init__()
        self.q_proj = nn.Linear(q_shape, config['attention_dim'])
        self.kv_proj = nn.Linear(k_shape + v_shape, config['attention_dim'] * 2)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, query, key, value):
        q = self.q_proj(query)
        k = self.q_proj(key)
        v = self.q_proj(value)
        x = torch.cat([q, k, v], dim=-1)
        x = self.kv_proj(x)
        x = self.dropout(x)
        return x

# Initializing the model
m = Model(
    q_shape=256,
    k_shape=512,
    v_shape=1024,
    dropout=config.get('attention_dropout_probs', 0)
)

# Inputs to the model
query = torch.randn(1, 3, 256)
key = torch.randn(1, 3, 256)
value = torch.randn(1, 3, 1024)
