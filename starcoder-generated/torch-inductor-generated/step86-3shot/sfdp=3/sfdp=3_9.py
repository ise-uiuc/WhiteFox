
class Model(torch.nn.Module):
    def __init__(self, in_channels, num_attention_heads, attention_head_size, out_channels, num_spatial_relations, dropout_p):
        super().__init__()
        self.attention_head_size = attention_head_size
        self.num_attention_heads = num_attention_heads
        self.matmul1 = torch.nn.Linear(in_channels, num_attention_heads * attention_head_size)
        self.num_spatial_relations = num_spatial_relations
        self.weight1 = torch.nn.Parameter(torch.randn(2 * num_spatial_relations, num_attention_heads, 1))
        self.weight2 = torch.nn.Parameter(torch.randn(1, 1, 1, 2 * num_spatial_relations))
        self.matmul2 = torch.nn.Linear(num_attention_heads * attention_head_size, out_channels)
        self.dropout_p = dropout_p
 
    def transpose_for_scores(self, x1):
        new_x1_shape = x1.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x1 = torch.reshape(x1, new_x1_shape)
        return x1.permute(0, 2, 1, 3)
 
    def forward(self, x1, x2, x3):
        bz, seq_len, channel = tuple(x1.shape)
        _, seq_len2, _ = tuple(x2.shape)
        x1 = self.transpose_for_scores(self.matmul1(x1))
        scale_factor = 1 / math.sqrt(self.attention_head_size)
        attention_logits = torch.matmul(x3.reshape(-1, self.num_spatial_relations, 2), self.weight1)
        attention_logits = torch.add(attention_logits, x2.reshape(seq_len, 1, -1) * self.weight2)
        attention_logits = torch.reshape(attention_logits, (-1, seq_len, seq_len2, 2 * self.num_spatial_relations)) # [bz, seq_len, seq_len2, 25]
        attention_logits = attention_logits.permute(0, 3, 1, 2) # [bz, 25, seq_len, seq_len2]
        attention_logits = torch.matmul(x1, attention_logits)
        attention_logits = torch.reshape(attention_logits, (-1, seq_len, seq_len2, self.num_attention_heads))
        attention_logits = attention_logits * scale_factor
        attention_logits = attention_logits.permute(0, 2, 3, 1)
        attention_logits = attention_logits.reshape(-1, seq_len, seq_len2)
        attention_weights_dropout = torch.nn.functional.dropout(F.softmax(attention_logits, dim=-1), p=self.dropout_p)
        x3_dropout = self.matmul2(attention_weights_dropout)
        return x3_dropout

# Initializing the model
in_channels = 32
num_attention_heads = 16
attention_head_size = in_channels // num_attention_heads
out_channels = 64
num_spatial_relations = 25
dropout_p = 0.1
m = Model(in_channels, num_attention_heads, attention_head_size, out_channels, num_spatial_relations, dropout_p)

# Inputs to the model
x1 = torch.randn(4, 32, 32)
x2 = torch.randn(32, 16)
x3 = torch.randn(4, 16, 32)
