
class Model(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads, query_len, key_len, dropout_p, attn_mask=None):
        super().__init__()
        self_attn = []
        for _ in range(atten_layer_num):
            self_attn.append(AttentionLayer(embedding_dim, num_heads, dropout_p, attn_mask))
        self.self_attn_layer_stack = torch.nn.Sequential(*self_attn)

    def forward(self, query, key, value):
        return self.self_attn_layer_stack(query, key, value)

# Initializing the model
embedding_dim = 256
num_heads = 8
query_len, key_len = 8 * 8, 16 * 16
dropout_p = 0.1
attn_mask = torch.randn([query_count, 1, key_len], dtype=torch.float32)
m = Model(embedding_dim, num_heads, query_len, key_len, dropout_p, attn_mask)

# Inputs to the model
query = torch.randn(query_count, num_heads, query_len, embedding_dim)
key = torch.randn(query_count, num_heads, key_len, embedding_dim)
value = torch.randn(query_count, num_heads, key_len, embedding_dim)
