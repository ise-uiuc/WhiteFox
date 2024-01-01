
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, scale_factor, dropout_p):
        _sum = query + key
        _sum2 = torch.nn.functional.relu(_sum)
        _sum3 = torch.nn.functional.softmax(_sum2, dim=-1)
        _sum4 = torch.nn.functional.dropout(_sum3, p=dropout_p)
        v1 = torch.matmul(_sum4, value)
        return v1

# Initializing the model
m = Model()

# Input to the model
query = torch.randn(3, 8, 4)
key = torch.randn(3, 8, 8)
value = torch.randn(3, 8, 8)
scale_factor = torch.rand(1)
dropout_p = 0.3
