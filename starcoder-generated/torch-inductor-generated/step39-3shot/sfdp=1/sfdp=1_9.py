
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout_prob = dropout_p
        self.total_key_depth = self.num_heads * self.embedding_dim
        self.total_value_depth = self.num_heads * self.embedding_dim
        self.input_depth = total_key_depth + total_value_depth
        self.qkv_transform = torch.nn.Linear(total_key_depth, 3 * self.total_value_depth)
 
    def forward(self, q):
        qkv = self.qkv_transform(q)
        qkv = torch.chunk(qkv, 3, dim=-1)
        q, k, v = qkv[0], qkv[1], qkv[2]
        return q, k, v

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 1, 256)
__output1__, __output2__, __output3__ = m(q)

