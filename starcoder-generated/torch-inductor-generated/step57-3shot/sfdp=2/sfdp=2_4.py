
class Model(torch.nn.Module):
    def __init__(self, hidden_size=2048, num_heads=2, dropout_p=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.qkv = torch.nn.Linear(hidden_size, 3 * hidden_size)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.proj = torch.nn.Linear(hidden_size, hidden_size)
        self.scaling = (hidden_size // num_heads) ** -0.5
 
    def forward(self, query, key, value):
        qkv = self.qkv(query).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        k = torch.transpose(k, -1, -2)
        scaled_qk = torch.matmul(q, k) * self.scaling
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v)
        return self.proj(output.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3)
 
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 64, 1, 2048)
key = torch.randn(1, 64, 1, 2048)
value = torch.randn(1, 64, 1, 2048)
