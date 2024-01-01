
class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_weight = torch.nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        self.k_weight = torch.nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))

    def forward(self, query, key, value, attention_mask):
        n = torch.matmul(query, self.q_weight)
        n = torch.matmul(key, self.k_weight)
        n = n / math.sqrt(self.q_weight.size(0))
        n = n + attention_mask

        n = torch.softmax(n, dim=-1)

        n = torch.matmul(n, value)

        return n

# Initializing the model
config = TransformerConfig(32)
m = Model(config)

# Inputs to the model
query = torch.randn(2, 32, 16)
key = torch.randn(2, 32, 16)
value = torch.randn(2, 32, 16)
attention_mask = torch.randn(2, 1, 1, 16)
