
class Model(torch.nn.Module):
    def __init__(self, num_heads=1, embedding_dim=16, dropout_p=0.6):
        super().__init__()
        self.num_heads = num_heads
        assert embedding_dim * num_heads % 8 == 0  # assert modulo 8
        self.depth = embedding_dim * num_heads // 8
        self.wq = torch.nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.wk = torch.nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.wv = torch.nn.Linear(embedding_dim, embedding_dim * num_heads)
        self.dense = torch.nn.Linear(embedding_dim * num_heads, embedding_dim)
 
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
 
    def forward(self, x1):
        batch_size = x1.shape[0]
        q = self.wq(x1)  # shape (batch_size, len_query, depth)
        k = self.wk(x1)  # shape (batch_size, len_key, depth)
        v = self.wv(x1)  # shape (batch_size, len_value, depth)
        q = self.split_heads(q, batch_size)  # shape (batch_size, num_heads, len_query, depth)
        k = self.split_heads(k, batch_size)  # shape (batch_size, num_heads, len_key, depth)
        v = self.split_heads(v, batch_size)  # shape (batch_size, num_heads, len_value, depth)
        # Compute scaled dot product attention
        scaled_attention = scaled_dot_product_attention(
            q, k, v, self.depth  # arguments are unpacked here
        )
        scaled_attention = scaled_attention.permute(
            0, 2, 1, 3
        )  # shape (batch_size, len_query, num_heads, depth)
        concat_attention = scaled_attention.reshape(
            batch_size, -1, self.depth * self.num_heads
        )  # shape (batch_size, len_query, embedding_dim)
        output = self.dense(concat_attention)  # shape (batch_size, len_query, embedding_dim)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(32, 128, 768)
