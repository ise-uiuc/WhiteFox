
class Attention(torch.nn.Module):
    def __init__(self, hidden_dim, dropout_p):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_dropout = torch.nn.Dropout(dropout_p)

    def forward(self, value, key, query):
        batch_size_v = value.size(0)
        batch_size_k = key.size(0)
        query = query.unsqueeze(1).expand(batch_size_v, batch_size_k, self.hidden_dim)
        key = key.unsqueeze(0).expand_as(query)
        values = value.unsqueeze(0).expand_as(query)
        weights = torch.sum(F.tanh(query + key) * values, dim=2)
        weights = self.attention_dropout(torch.softmax(weights, dim=1))
        return weights

# Initializing the model
hidden_dim = 32
dropout_p = 0.2

__input__ = [hidden_dim * 8, hidden_dim * 8, hidden_dim * 2]
m = Attention(hidden_dim, dropout_p)
