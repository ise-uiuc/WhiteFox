
class Model(torch.nn.Module):
    def __init__(self, dropout_p, num_heads, num_heads_per_partition, num_partitions):
        super().__init__()
        self.dropout_p = dropout_p
        self.num_heads = num_heads
        self.num_partitions = num_partitions
        self.num_heads_per_partition = num_heads_per_partition
        self.scale_factor = 1 / (np.sqrt(latent_dim) * num_heads_per_partition)
        self.to_k = nn.Parameter(torch.randn(num_heads * num_heads_per_partition, latent_dim, latent_dim))
        self.to_q = nn.Parameter(torch.randn(num_heads * num_heads_per_partition, latent_dim, latent_dim))
        self.to_v = nn.Parameter(torch.randn(num_heads * num_heads_per_partition, latent_dim, latent_dim))
 
    def forward(self, keys, queries, values, training=False):
        keys = rearrange(keys, 'b n (h p) d -> (b h p) n d', h=self.num_heads, p=self.num_heads_per_partition)
        queries = rearrange(queries, 'b n (h p) d -> (b h p) n d', h=self.num_heads, p=self.num_heads_per_partition)
        values = rearrange(values, 'b n (h p) d -> (b h p) n d', h=self.num_heads, p=self.num_heads_per_partition)
        qk = torch.matmul(queries, torch.transpose(keys, -2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        attention = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = torch.matmul(attention, values)
        output = rearrange(output, '(b h p) n d -> b n (h p) d', b=1, h=self.num_heads, p=self.num_heads_per_partition)
        return output

# Initializes the model
m = Model(dropout_p, num_heads, num_heads_per_partition, num_partitions)

# Inputs to the model
keys = torch.randn(1, 16, latent_dim)
values = torch.randn(1, 16, latent_dim)
queries = torch.randn(1, 16, latent_dim)
