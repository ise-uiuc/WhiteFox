
class Model(torch.nn.Module):
    def __init__(self, num_heads, embed_dim, dropout_p):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = self.scaling ** -1
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

    def _reshape_to_batches(self, x):
        batch_size = x.size(0)
        head_dim = self.head_dim
        embed_dim = self.embed_dim
        num_heads = self.num_heads
        seq_length = x.size(1)
        x = x.reshape(batch_size, seq_length, num_heads, head_dim)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size * num_heads, seq_length, head_dim)
        return x

# Initializing the model
num_heads = 12
embed_dim = 768
dropout_p = 0.1
m = Model(num_heads, embed_dim, dropout_p)

# Inputs to the model
x1 = torch.randn(1, 10, embed_dim)
x2 = torch.randn(2, 10, embed_dim)
