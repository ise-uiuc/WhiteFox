
class Model(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_p=0.5):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = torch.nn.Linear(hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(hidden_size, self.all_head_size)

        self.dropout = torch.nn.Dropout(p=dropout_p)

    def transpose_for_scores(self, q, k, v):
        new_q_shape = q.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        q = q.view(*new_q_shape)
        k = k.view(*new_q_shape)
        v = v.view(*new_q_shape)

        return (q, k, v)

    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x1)
        v = self.value(x2)

        q, k, v = self.transpose_for_scores(q, k, v)
        output = torch.matmul(q, k.transpose(-2, -1))
        output = output.softmax(dim=-1)
        output = self.dropout(output)
        output = torch.matmul(output, v)

        output = output.view(*output.size()[:-2], output.size(-2), self.hidden_dim)

        return output

model = Model(
  hidden_size=12, num_attention_heads=4)
    
# Initializing the model
m = Model(hidden_size=12, num_attention_heads=4)

# Inputs to the model
x1 = torch.randn(2, 10, 12)
x2 = torch.randn(2, 15, 12)
