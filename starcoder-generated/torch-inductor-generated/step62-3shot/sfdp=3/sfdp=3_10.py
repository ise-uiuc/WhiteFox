
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        assert embedding_dim % num_heads == 0, 'embedding_dim should be divisible by num_heads'
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.padding_idx = padding_idx
        self.scale_factor = torch.tensor(math.sqrt(embedding_dim), dtype=torch.float)
        # Parameters of Multi-Headed Attention
        self.q_proj = nn.Linear(query_feature_size, num_heads * embedding_dim, bias=False)
        self.k_proj = nn.Linear(key_feature_size, num_heads * embedding_dim, bias=False)
        self.v_proj = nn.Linear(value_feature_size, num_heads * embedding_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * embedding_dim, output_feature_size, bias=False)

    def _split_heads(self, x, incremental_state=None):
        return x.view(x.size(0), x.size(1), self.num_heads, self.embedding_dim).transpose(1, 2)

    def _combine_heads(self, x, incremental_state=None):
        return x.transpose(1, 2).contiguous().view(x.size(0), -1, self.num_heads * self.embedding_dim)

    def forward(self, query, key, value, incremental_state=None, need_weights=False):
        # Multi-Headed Attention
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))
        q *= self.scale_factor
        x = ops.multi_head_attention_float_16(
            q, k, v, self.embedding_dim, self.num_heads, self.padding_idx,
            incremental_state, self.training,
            need_weights, self.scale_factor)
        x = self._combine_heads(x)
        x = F.linear(x, self.o_proj.weight, self.o_proj.bias)
        return x

# Initializing the model
m = Model()

# Inputs of the model
num_steps = 2
batch_size = 3
seq_len = 8
query_feature_size = 16
key_feature_size = 16
value_feature_size = 16    
output_feature_size = 24
padding_idx = -2
output_padding = 1

query = torch.randn(num_steps, batch_size, query_feature_size)
key = torch.randn(num_steps, batch_size, seq_len, key_feature_size)
value = torch.randn(num_steps, batch_size, seq_len, value_feature_size)
