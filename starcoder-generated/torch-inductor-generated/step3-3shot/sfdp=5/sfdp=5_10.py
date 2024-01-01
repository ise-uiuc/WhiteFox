
class Model(torch.nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = float(dropout)

    def forward(self, query, key, value):
        output = torch.softmax((query @ key.transpose(-2, -1)) / math.sqrt(query.size(-1)), dim=-1)
        output = torch.dropout(output, self.dropout, training=self.training)
        output = output @ value
        return output

# Initializing the model
m = Model(dropout=0.1)

# Inputs to the model
query = torch.randn(4, 16, 10)
key = torch.randn(4, 32, 10)
value = torch.randn(4, 32, 20)
