
class Model(torch.nn.Module):
    def __init__(self, query_hidden_size, key_hidden_size, value_hidden_size, num_heads, dropout):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(query_hidden_size, num_heads, dropout=dropout)
 
    def forward(self, queries, keys, values, inv_scale_factor):
        self.mha.in_proj_weight = torch.nn.Parameter(0.25 * (1 - inv_scale_factor) ** 0.5 * torch.eye(queries.shape[1]) / query**2)
        v1 = self.mha(queries, keys, values)
        return v1

# Initializing the model
m = Model(query_hidden_size, key_hidden_size, value_hidden_size, num_heads, dropout)

# Inputs to the model
torch.manual_seed(0)
mha_input_shape = (seq_length, bsz, embed_dim)
queries = torch.randn(mha_input_shape)
keys = torch.randn(mha_input_shape)
values = torch.randn(mha_input_shape)
inv_scale_factor = 1 / math.sqrt(embed_dim)
