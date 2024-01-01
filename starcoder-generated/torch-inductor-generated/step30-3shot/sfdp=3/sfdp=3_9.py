.
class ScaledDotProductAttention(nn.Module):
    def __init__(self, key_size, dropout_p=0.1):
        super().__init__()
        self.key_dim = key_size
        self.layer_norm = nn.LayerNorm(key_size)
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(key_size, key_size, bias=True)

    def forward(self, inputs, scale_factor):
        attention = torch.matmul(inputs, inputs.transpose(-2, -1))
        attention = attention * scale_factor
        softmax_attention = F.softmax(attention, dim=-1)
        dropout_attention = self.dropout(softmax_attention)
        output = torch.matmul(dropout_attention, inputs)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, key_dims, num_heads, dropout_ps=0.0):
        super().__init__()
        self.linear = nn.ModuleList([nn.Linear(key_dims, key_dims, bias=True) for _ in range(num_heads)])
        self.layer_norm = nn.ModuleList([nn.LayerNorm(key_dims) for _ in range(num_heads)])
        self.attention_scale_factors = nn.ParameterList([nn.Parameter(torch.ones(key_dims)) for _ in range(num_heads)])
        self.dropouts =nn.ModuleList([nn.Dropout(dropout_ps) for _ in range(num_heads)])
        self.output_linear = nn.Linear(key_dims * num_heads, key_dims, bias=True)
        self.num_heads = num_heads
 
    def forward(self, inputs):
        outputs = []
        for l in range(self.num_heads):
            query = self.linear[l](inputs)
            scale_factor = self.attention_scale_factors[l].unsqueeze(0).unsqueeze(1)
            transformed_query = self.layer_norm[l](query)
            attention = ScaledDotProductAttention(self.key_dims, dropout_p=0.1)(transformed_query, scale_factor)
            dropout_attention = self.dropouts[l](attention)
            head = dropout_attention.reshape(inputs.shape[:2] + (-1,))
            output = self.output_linear(head)
            outputs += [output]
        return torch.stack(outputs).mean(dim=0)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=30, embedding_dim=30, padding_idx=0)
        self.encoder_self_attention = MultiHeadAttention(key_dims=30, num_heads=10)
        self.dense = nn.Linear(hidden_size, 20)

    def forward(self, inputs):
        x = inputs.clone()
        x[:, 1:] += self.embeddings(x[:, :-1])
        x = self.encoder_self_attention(x)
        x = self.dense(x)
        return x

# Initializing the model.
model = Model()

# Inputs to the model.
input_ = torch.randint(30, (2, 20))
prev_h = torch.randn(1, 20)
