
class Model(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, layer_norm_eps):
        super().__init__()
        self.q = torch.nn.Linear(hidden_size, hidden_size)
        self.k = torch.nn.Linear(hidden_size, hidden_size)
        self.v = torch.nn.Linear(hidden_size, hidden_size)
        self.proj = torch.nn.Linear(hidden_size, hidden_size)
    
    def forward(self, q, k, v, mask):
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        q = q / math.sqrt(k.size(-1))
        q = q * mask
        k = k + q.masked_fill(~mask, float(-10000.0)).unsqueeze(-1)
        context_weight = torch.nn.Softmax(dim=-1)(k)
        context_weight = torch.nn.Dropout(self.dropout_prob)(context_weight)
        out = context_weight @ v
        return self.layer_norm(out + residual)

# Initializing the model
m = Model(hidden_size=1024,
          num_attention_heads=4,
          attention_probs_dropout_prob=0.1,
          layer_norm_eps=1e-5)

# Inputs to the model
q = torch.randn(1, num_attention_heads, hidden_size)
k = torch.randn(1, num_attention_heads, hidden_size)
v = torch.randn(1, num_attention_heads, hidden_size)
mask = torch.randn((1, 1, num_attention_heads)) < 0
