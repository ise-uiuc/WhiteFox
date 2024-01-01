
class Model(torch.nn.Module):
    def forward(self, query, key, value, key_padding):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + key_padding
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, True)
        output = attn_weight @ value
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 16, 32, 64)
key = torch.randn(1, 16, 64, 64)
value = torch.randn(1, 16, 64, 64)
key_padding = torch.randn_like(query, requires_grad=False)

