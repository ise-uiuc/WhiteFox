
class Model(torch.nn.Module):
    def __init__(self, vocab_size, num_attention_heads, hidden_size, hidden_dropout_prob):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(hidden_size, num_attention_heads, hidden_dropout_prob)
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.intermediate = torch.nn.Linear(hidden_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x1, x2):
        v1 = self.attention(x1, x2, x2)[0]
        v2 = self.dense(v1)
        v3 = v2 * 0.5
        v4 = self.intermediate(v3)
        v5 = self.output(v4)
        return v5

# Initializing the model
m = Model(vocab_size=50, num_attention_heads=50, hidden_size=40, hidden_dropout_prob=0)

# Inputs to the model
x1 = torch.randn(2, 3, 50, 40)
x2 = torch.randn(2, 4, 50, 40)
