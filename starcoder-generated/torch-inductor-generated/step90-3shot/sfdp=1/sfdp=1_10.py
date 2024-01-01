
class Model(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_size = self.hidden_size // self.num_attention_heads 
        self.scale_factor = self.head_size ** -0.5
        self.qkv = torch.nn.Linear(self.hidden_size, 3 * hidden_size)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x1):
        qkv = self.qkv(x1).reshape(1, 3, self.hidden_size, self.num_attention_heads).transpose(1, 2)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scaled_qk = torch.matmul(query, key.transpose(-2, -1)) * self.scale_factor
        softmax_qk = scaled_qk.softmax(-1)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, value)
        return output, query, value

# Initializing the model
m = Model(1024, 8)

# Inputs to the model
x1 = torch.randn(1, 1024)
