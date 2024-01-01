
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 10
        self.num_heads = 2
        self.num_layers = 4
        self.query_projection = torch.nn.Linear(self.embedding_dim, self.num_heads * self.embedding_dim)
        self.key_projection = torch.nn.Linear(self.embedding_dim, self.num_heads * self.embedding_dim)
        self.value_projection = torch.nn.Linear(self.embedding_dim, self.num_heads * self.embedding_dim)
        self.out_projection = torch.nn.Linear(self.num_heads * self.embedding_dim, self.embedding_dim)
 
    def get_output_size(self):
        return [self.num_heads * self.embedding_dim]
 
    def get_attention_map(self, input_tensor):
        return input_tensor
 
    def forward(self, query_tensor, key_tensor, value_tensor, attn_mask):
        query_proj = self.query_projection(query_tensor)
        key_proj = self.key_projection(key_tensor)
        value_proj = self.value_projection(value_tensor)
 
        # split
        size = query_proj.size()
        query_proj = query_proj.view(*size[:-1], self.num_heads, self.embedding_dim)
        key_proj = key_proj.view(*size[:-1], self.num_heads, self.embedding_dim)
        value_proj = value_proj.view(*size[:-1], self.num_heads, self.embedding_dim)
 
# Initializing the model
m = Model()

# Inputs to the model
query_tensor = torch.randn(1, 2, 10)
key_tensor = torch.randn(1, 4, 10)
value_tensor = torch.randn(1, 4, 10)
attn_mask = torch.randn(1, 1, 2, 4)
size = (*query_tensor.size()[:-1], *(m.get_output_size()))
__output__query_proj = m.query_projection(query_tensor)
__output__key_proj = m.key_projection(key_tensor)
__output__value_proj = m.value_projection(value_tensor)
__output__attn_mask = attn_mask
# attention map generation
__output__attn_map = m.get_attention_map(torch.cat([query_tensor, key_tensor, value_tensor], dim=-1))
__output__scaled = torch.softmax((__output__query_proj @ __output__key_proj.transpose(-2, -1) / math.sqrt(__output__query_proj.size(-1))), dim=-1) + __output__attn_mask
__output__weighted_sum = (__output__scaled @ __output__value_proj).view(*size)
__output__context_sensitive_representation = m.out_projection(__output__weighted_sum)

