
class Model(torch.nn.Module):
    def __init__(self, input_dim, value_dim, key_dim=None, num_heads=8, dropout_p=0.2):
        super().__init__()
        key_dim = input_dim if key_dim is None else key_dim
        self.scale_factor = 1 / (key_dim ** 0.5)
        # TODO: please use `torch.nn.MultiheadAttention` to implement the attention layer. Hint: you need to specify the default value of parameter `batch_first` as `True`. 
        self.attention = torch.nn.MultiheadAttention(input_dim, num_heads, dropout=dropout_p, batch_first=True)
 
    def forward(self, x1, batch):
        q = k = self.scale_factor * self.attention.in_proj_weight[:input_dim]
        v = self.attention.in_proj_weight[input_dim:]
        x2 = self.attention(q, k, v)[0]
        return x2

# Initializing the model
input_dim = vocab_size
value_dim = 64
num_heads = 8
dropout_p = 0.2
m = Model(input_dim, value_dim, batch=num_heads, dropout_p=dropout_p)

# Inputs to the model
x1 = torch.randn(2, 16, input_dim)
batch = [x1.shape[0], x1.shape[1]]
x2 = m(x1, batch)

