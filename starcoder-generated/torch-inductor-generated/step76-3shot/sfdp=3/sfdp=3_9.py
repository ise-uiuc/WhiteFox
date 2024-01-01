
class Model(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, dropout_p):
        super().__init__()
        self.attention = MultiHeadedAttention(input_dim=input_dim, hidden_size=hidden_size)
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, x1, x2):
        q, k, v = x1, x2, x2
        scale_factor = 1 / math.sqrt(q.shape[-1])
        v, attention_weights = self.attention(q, k, v), None
        v = self.dropout(v)
        return v

# Initializing the model
m = Model(input_dim=64, hidden_size=128, dropout_p=0.05)

# Inputs to the model
x = torch.randn(1, 10, 64)
