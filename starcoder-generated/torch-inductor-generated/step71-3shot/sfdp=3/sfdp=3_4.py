
class Model(torch.nn.Module):
    def __init__(self, num_heads, head_dim, dropout_p):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, num_heads * head_dim)
        self.linear2 = torch.nn.Linear(4, num_heads * head_dim)
        self.linear3 = torch.nn.Linear(4, num_heads * head_dim)
        self.scale_factor = num_heads ** -0.5
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, x1, x2, x3):
        v1 = self.linear1(x1)
        v2 = self.linear2(x2)
        v3 = self.linear3(x3)
        v4 = torch.matmul(v1, v2.transpose(-2, -1))
        v5 = v4 * self.scale_factor
        v6 = torch.nn.functional.softmax(v5, dim=-1)
        v7 = self.dropout(v6)
        v8 = v7.matmul(v3)
        return v8

# Initializing the model
dropout_p = 0.4
m = Model(4, 8, dropout_p)

# Inputs to the model
x1 = torch.randn(1, 16, 4)
x2 = torch.randn(1, 16, 4)
x3 = torch.randn(1, 16, 4)
