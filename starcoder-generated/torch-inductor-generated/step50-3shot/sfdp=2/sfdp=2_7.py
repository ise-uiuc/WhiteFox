
class Model(torch.nn.Module):
    def __init__(self, num_heads=8, query_size=64):
        super().__init__()
        self.query = nn.Linear(query_size, query_size)
        self.key = nn.Linear(query_size, query_size)
        self.value = nn.Linear(query_size, query_size)
        self.inv_scale_factor = math.sqrt(query_size)
        self.dropout_p = 0.5
        self.num_heads = num_heads
 
    def forward(self, x1, x2, x3):
        v1 = self.query(x1)
        v2 = self.key(x2)
        v3 = self.value(x3)
        v4 = torch.matmul(v1, v2.transpose(-2, -1))
        v5 = v4.div(self.inv_scale_factor)
        v6 = v5.softmax(dim=-1)
        v7 = torch.nn.functional.dropout(v6, p=self.dropout_p)
        v8 = torch.matmul(v7, v3)
        return v8

# Initializing the model
model = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 14, 14)
x2 = torch.randn(1, 64, 14, 14)
x3 = torch.randn(1, 64, 14, 14)
