
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads, bias, add_bias_kv):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_size = int(hidden_size / num_attention_heads)
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.scale = 1 / math.sqrt(self.attention_head_size)
        self.bias = bias
        self.add_bias_kv = add_bias_kv

        self.q_proj = torch.nn.Linear(hidden_size, self.all_head_size)
        self.kv_proj = torch.nn.Linear(hidden_size, 2 * self.all_head_size)
        if self.add_bias_kv:
            self.k_bias = torch.nn.Parameter(torch.zeros(bias.size()))
            self.v_bias = torch.nn.Parameter(torch.zeros(bias.size()))
        else:
            self.bias = torch.nn.Parameter(torch.zeros(bias.size()))

    def forward(self, query, key, value):
        