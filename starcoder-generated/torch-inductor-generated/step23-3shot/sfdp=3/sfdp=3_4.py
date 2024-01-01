
class Model(torch.nn.Module):
    def __init__(self, num_heads: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        head_dim = embedding_dim // num_heads
        self.head_dim = head_dim
        #... code omitted...
     
 
    def forward(self, query, key, value, mask=None):
        #... code omitted...
        # batch_size: batch_size,
        # num_heads: self.num_heads,
        # head_dim: self.head_dim,
        # key_len: key.size(-2),
        # drop_prob: self.dropout
        #... code omitted...

# Inputs to the model
query = torch.randn(2, 3, 62, 16)
key = torch.randn(2, 4, 74, 16)
value = torch.randn(2, 4, 74, 16)
