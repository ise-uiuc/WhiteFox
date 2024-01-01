
class ModelQueryKeyDotProduct(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.key = torch.nn.Linear(config.hidden_size, config.hidden_size)
 
    def forward(self, x1):
        q = self.query(x1)
        k = self.key(x1)
        m = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        return m

# Initializing the model
m = ModelQueryKeyDotProduct(config)

# Inputs to the model
x1 = torch.randn(1, 2, 3)
