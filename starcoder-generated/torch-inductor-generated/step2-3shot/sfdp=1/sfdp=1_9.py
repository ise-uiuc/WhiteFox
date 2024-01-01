
class Model(torch.nn.Module):
    def __init__(self, dim_head, num_heads, dropout_p):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.query = torch.nn.Linear(1024, dim_head * num_heads)
        self.key = torch.nn.Linear(1024, dim_head * num_heads)
        self.value = torch.nn.Linear(1024, dim_head * num_heads)
        self.dropout_p = dropout_p
 
    def forward(self, x1):
        qk = self.query(x1)
        qk = self.key(qk)
        qk_scaled = (qk * self.scale).softmax(dim=-1)
        return qk_scaled.matmul(value)

# Initializing the model
m = Model(dim_head=16, num_heads=64, dropout_p=0.9)

# Inputs to the model
x1 = torch.randn(1, 1024)
