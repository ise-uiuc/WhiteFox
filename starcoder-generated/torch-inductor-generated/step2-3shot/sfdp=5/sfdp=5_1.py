
class Model(torch.nn.Module):
    def __init__(self, n_head, d_model, dropout_p):
        super().__init__()
        self.inner_dim = n_head * d_model 
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 / math.sqrt(self.inner_dim)
        v3 = v2 + x3
        v4 = torch.softmax(v3, dim=-1)
        v5 = v4.unsqueeze(1) # Reshape the attention weights to 4, 1,heads,query_length, key_length
        v6 = self.dropout(v5)
        v7 = torch.matmul(v6, x3) # output = attn_weight @ value
        v8 = v7.transpose(1, 2).contiguous().view(v7.size(0), -1, self.n_head * self.d_model)
        return v8

# Initializing the model
m = Model(n_head=4, d_model=64, dropout_p=0.5)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 4, 64, 64)
