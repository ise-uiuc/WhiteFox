
class Model(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, attn_dropout, output_dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.scale_factor = math.sqrt(self.head_size)
        self.attn_dropout = torch.nn.Dropout(p=attn_dropout)
        self.output_dropout = torch.nn.Dropout(p=output_dropout)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.attn_dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(hidden_size, num_heads, attn_dropout, output_dropout)

# Inputs to the model
query, key, value = torch.randn(64, 3, hidden_size), 
                  torch.randn(64, 3, hidden_size), 
                  torch.randn(3, hidden_size)
