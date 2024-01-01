
class Model(torch.nn.Module):
    def __init__(self, num_attention_heads, hidden_dim):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_dim
    
        self.key_projection = torch.nn.Linear(hidden_dim, num_attention_heads * hidden_dim)
        self.value_projection = torch.nn.Linear(hidden_dim, num_attention_heads * hidden_dim)
        self.query_projection = torch.nn.Linear(hidden_dim, num_attention_heads * hidden_dim)
        self.out_projection = torch.nn.Linear(hidden_dim, num_attention_heads * hidden_dim)
    
    def transpose_for_scores(self, x1):
        new_x1_shape = x1.size()[:-1] + (self.num_attention_heads, -1)
        x1 = x1.view(*new_x1_shape)
        #[0:1, 2, 3][1, 2, 3]   (num_attention_heads, batch_size, seq_len, hidden_size)
        x1 = x1.permute(0, 2, 1, 3)
        return x1
        
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        query = torch.matmul(query, self.key_projection.transpose(0, 1))
        key = torch.matmul(torch.transpose(key, 0, 1), self.key_projection.transpose(0, 1))
        value = torch.matmul(torch.transpose(value, 0, 1), self.value_projection.transpose(0, 1))
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
m = Model(4, 8)

# Inputs to the model
query = torch.randn(2, 3, 4, 8)
key = torch.randn(2, 6, 4, 8)
value = torch.randn(2, 6, 4, 8)
inv_scale_factor = 3.14
dropout_p = 0.05
