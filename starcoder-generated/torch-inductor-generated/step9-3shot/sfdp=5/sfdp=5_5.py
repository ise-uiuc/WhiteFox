
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_linear = torch.nn.Linear(64, 64, bias=False)
        self.key_linear = torch.nn.Linear(64, 64, bias=False)
        self.linear = torch.nn.Linear(64, 64, bias=False)
        self.dropout = torch.nn.Dropout(0.1)
 
    def forward(self, query, key):
        query = self.query_linear(query)
        key = self.key_linear(key)
        output = query.size()
        attn_weight = torch.softmax((query + key).reshape(output[0], output[1], -1), dim=-1)
        print(attn_weight.size())
        attn_weight = self.dropout(attn_weight)
        output = attn_weight.size()
        output = (attn_weight.reshape(output[0], output[1], 1, 1) * value).sum(dim=1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 4, 8)
key = torch.randn(1, 4, 8, 8)
value = torch.randn(1, 4, 8, 8)
