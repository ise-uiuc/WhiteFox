
class FeedForwardNetwork(modules.torch.Child):
    def __init__(self, dim_in, dim_hidden, dropout, activation):
        super().__init__()
        self.layer1 = modules.torch.Linear(dim_in, dim_hidden)
        self.act = modules.torch.Activation(activation)
        self.dropout = modules.torch.Dropout(dropout)
        self.layer2 = modules.torch.Linear(dim_hidden, dim_in)
 
    def forward(self, input_tensor):
        output = self.layer1(input_tensor)
        output = self.act(output)
        output = self.dropout(output)
        output = self.layer2(output)
        return output

class MultiHeadAttention(modules.torch.Child):
    def __init__(self, dim, out_dim, num_heads, dropout):
        super().__init__()
        
        if dim % num_heads!= 0:
            raise ValueError('Number of attention heads must be a factor of the dimensionality of the hidden vector.')
 
        dim_head = dim // num_heads
        self.qkv = modules.torch.ChildList(modules.torch.Linear(dim, dim * 3))
        self.out = modules.torch.Child(FeedForwardNetwork, dim=out_dim, dim_hidden=dim_head * dim, dropout=dropout, activation='relu')
        self.num_heads = num_heads
        self.scale_factor = math.sqrt(dim_head)
    
    def forward(self, input_1, input_2):
        queries = self.qkv[0](input_1)
        keys = self.qkv[1](input_1)
        values = self.qkv[2](input_1)
        
        query, key, value = torch.split(queries, queries.shape[-1] // self.num_heads, dim=-1)
        key, value = torch.split(keys, keys.shape[-1] // self.num_heads, dim=-1)
        query, key, value = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (query, key, value))
        
        qk = torch.einsum('b h n d, b h n d -> b h n d', query, key)
        scaled_qk = qk / self.scale_factor
 
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
 
        output = torch.einsum('b h n d, b h n d -> b h n d', dropout_qk, value)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.out(output)
        return output

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = MultiHeadAttention(dim=64, out_dim=64, num_heads=4, dropout=dropout_p)
        self.attn2 = MultiHeadAttention(dim=64, out_dim=64, num_heads=4, dropout=dropout_p)
        self.norm = torch.nn.LayerNorm(64)
        self.dense1 = torch.nn.Linear(64, 64)
        self.dense2 = torch.nn.Linear(64, 64)
    
    def forward(self, x):
        input_a = x + self.attn1(x, x)
        input_a = input_a + self.attn2(input_a, input_a)
        x = self.norm(x + input_a)
        x = self.dense1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, p=dropout_p)
        x = self.dense2(x)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(input_shape)
