
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, attention_mask):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1 / math.sqrt(query.size(-1))
        v3 = v2 + attention_mask
        v4 = torch.softmax(v3, dim=-1)
        v5 = torch.matmul(v4, value)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 60, 64)
key = torch.randn(1, 8, 60, 64)
value = torch.randn(1, 8, 60, 64)
attention_mask = torch.Tensor([[0, 0, 0, -10000, -10000],
                                  [0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0]])
