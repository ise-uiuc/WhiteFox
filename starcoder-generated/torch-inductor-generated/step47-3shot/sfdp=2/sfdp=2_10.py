
class Model(torch.nn.Module):
    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.channels_size = channels
        self.head_size = channels // num_heads
        self.linear_q = torch.nn.Linear(channels, channels, bias=True)
        self.linear_k = torch.nn.Linear(channels, channels, bias=True)
        self.linear_v = torch.nn.Linear(channels, channels, bias=True)
        self.scaling_factor = np.sqrt(self.head_size)
 
    def forward(self, x1, x2, x3):
        q_vector = self.linear_q(x1)
        k_vector = self.linear_k(x2)
        v_vector = self.linear_v(x3)
        q_head, k_head, v_head = self.compute_qkv_head(q_vector), self.compute_qkv_head(k_vector), self.compute_qkv_head(v_vector)
        attention = torch.matmul(q_head, (k_head.transpose(-2, -1)))
        scaled_attention = attention.div(self.scaling_factor)
        softmax_attention = scaled_attention.softmax(dim=-1)
        dropout_attention = torch.nn.functional.dropout(softmax_attention, p=0.5)
        output = torch.matmul(dropout_attention, v_head)
        output = self.combine_heads(output)
        return output
 
    def compute_qkv_head(self, x: torch.Tensor):
        batch_size, seq_length, _ = x.shape
        qkv_head = torch.reshape(x, (batch_size, seq_length, self.num_heads, self.head_size))
        qkv_head = torch.transpose(qkv_head, 1, 2)
        qkv_head = torch.reshape(qkv_head, (batch_size, self.num_heads, seq_length, self.head_size))
        return qkv_head
 
    def combine_heads(self, x: torch.Tensor):
        batch_size, _, seq_length, _ = x.shape
        x = torch.transpose(x, 1, 2)
        x = torch.reshape(x, (batch_size, seq_length, self.channels_size))
        return x

# Initializing the model with num_heads = 2
m = Model(3, 2)

# Inputs to the model
x1 = torch.randn(16, 3, 8)
x2 = torch.randn(16, 3, 8)
x3 = torch.randn(16, 3, 8)
