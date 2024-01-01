
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter([])
        self.key = torch.nn.Parameter([])
        self.value = torch.nn.Parameter([])
    
    def forward(self, x1):
        qk = self.query @ self.key.transpose(-2, -1) / math.sqrt(32)
        qk += attn_mask # Masks will be added in the next iteration
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weights, dropout_p, True)
        output = attn_weight @ value
        return output

# Initializing the model
m = Model(query, key, value)

# Inputs to the model
x1 = torch.randn(1, 32, 32)
