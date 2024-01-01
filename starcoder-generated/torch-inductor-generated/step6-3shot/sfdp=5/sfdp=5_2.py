
class Model(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_p):
        super().__init__()
        self.d_model = hidden_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.k_dim = d_model & (hidden_size // num_heads)
        self.v_dim = d_model & (hidden_size // num_heads)
 
        query_head = nn.Linear(d_model, hidden_size)
        key_head = nn.Linear(d_model, hidden_size)
        value_head = nn.Linear(d_model, hidden_size)
        self.attention = Attention(query_head, key_head, value_head)
 
    def forward(self, value, query):
        input_dim = value.size(2)
        q_k = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        q_k = q_k + self.attn_mask
        attn_weight = torch.softmax(q_k, dim=-1)
        attn_weight = torch.dropout(attn_weight, p=self.dropout_p, training=True)
        output = attn_weight @ value

# Initializing the model
model = Model(hidden_size=256, num_heads=8, dropout_p=0.1)

# Inputs to the model
value = torch.randn(1, 8, 256, 256)
query = torch.randn(1, 8, 256, 256)
