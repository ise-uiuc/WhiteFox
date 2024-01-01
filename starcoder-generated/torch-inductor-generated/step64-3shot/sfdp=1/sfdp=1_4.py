
class Model(torch.nn.Module):
    def __init__(self, num_heads=8, query_dims=32, key_dims=32, dropout_rate=0.1, scale_factor=1. / np.sqrt(query_dims)):
        super().__init__()
        self.num_heads = num_heads
        self.query = torch.nn.Linear(query_dims, num_heads * query_dims, bias=False)
        self.key = torch.nn.Linear(key_dims, num_heads * key_dims, bias=False)

        self.scale_factor = scale_factor
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x1, x2):
        q = self.query(x1).view(x1.size(0), x1.size(1), self.num_heads, -1)
        k = self.key(x2).view(x2.size(0), x2.size(1), self.num_heads, -1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1))

        scaled_logits = logits.div(self.scale_factor)

        softmax_logits = F.softmax(scaled_logits, dim=-1)

        dropout_logits = self.dropout(softmax_logits)

        return dropout_logits.matmul(v)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 16, 32)
x2 = torch.randn(3, 32, 64)
