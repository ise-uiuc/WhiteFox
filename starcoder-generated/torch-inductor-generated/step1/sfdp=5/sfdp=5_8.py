
class Model(torch.nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.query_net =...  # TODO
        self.key_net =...  # TODO
        self.attn_mask =...  # TODO
        self.dropout = torch.nn.Dropout(p=dropout_p, inplace=True)
 
    def forward(self, query_input, key_input, value_input):
        query = self.query_net(query_input)
        key = self.key_net(key_input)
        attn_scores = ((query @ key.transpose(-2, -1)) / math.sqrt(query.size(-1))) + self.attn_mask
        attn_weights = self.dropout(torch.nn.Softmax(dim=-1)(attn_scores))
        output = (self.attn_mask @ value_input) * attn_weights
        return output

# Initializing the model
m = Model(0.2)

# Inputs to the model
query_input = torch.randn(1, 2, 8)
key_input = torch.randn(1, 4, 8)
value_input = torch.randn(1, 4, 8)

