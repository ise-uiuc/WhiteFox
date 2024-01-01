
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.2
        self.scale_factor = hidden_dim ** -1
        self.project1 = nn.Linear(input_dim, input_dim)
        self.project2 = nn.Linear(input_dim, hidden_dim)

    def forward(self, query, key_value, mask=None):
        query = self.project1(query)
        qkv = self.project2(key_value)
        dot = torch.matmul(qkv, query.transpose(-2, -1))
        if mask is not None:
            dot = dot.masked_fill(mask == 0, -1e9)
        attn = F.softmax(dot * self.scale, dim=-1)
        attn = F.dropout(attn, self.dropout_p)

        output = torch.matmul(attn, qkv)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.rand(64, 32, input_dim)
key_value = torch.rand(64, 32, input_dim)
input_mask = torch.zeros(64, 32).to(torch.bool)
