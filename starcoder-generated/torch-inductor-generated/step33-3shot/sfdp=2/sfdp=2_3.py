
class Model(torch.nn.Module):
    def __init__(self, qkv_input_dim, seq_length, num_heads, head_dim, dropout_p=0.2):
        super().__init__()
        attention = torch.nn.Linear(qkv_input_dim, seq_length*num_heads*head_dim, bias=True)
        self.attention = MultiheadAttentionLayer(attention, seq_length, num_heads, head_dim, dropout_p=dropout_p)

    def forward(self, query, key, value):
        return self.attention(query, key, value)
