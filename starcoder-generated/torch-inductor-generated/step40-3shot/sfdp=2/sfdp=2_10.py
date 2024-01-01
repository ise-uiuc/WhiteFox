
class Model(torch.nn.Module):
    def __init__(self, input_shape, num_heads=1, dropout_p=0.5):
        super().__init__()
        self.num_heads = num_heads
        inner_dim = input_shape[-1]
        self.qkv = torch.nn.Linear(inner_dim, 3 * inner_dim)
        self.dropout_p = dropout_p
        self.scale_factor = torch.sqrt(torch.FloatTensor([inner_dim // num_heads]))
 
    def forward(self, query, key, value):
        qkv = self.qkv(query).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.num_heads), qkv)
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = 1 / self.scale_factor
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        output = rearrange(output, '(b h) n d -> b n (h d)', h=self.num_heads)
        return output

# Initializing the model
m = Model((4, 8), num_heads=2)

# Inputs to the model
query = torch.randn(4, 2)
key = torch.randn(2, 4)
value = torch.randn(2, 4)
