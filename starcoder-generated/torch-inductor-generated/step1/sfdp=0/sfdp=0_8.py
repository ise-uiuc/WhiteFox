
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Linear(16, 16)
        self.query = torch.nn.Linear(16, 16)
        self.value = torch.nn.Linear(16, 16)
 
    def forward(self, query, key, value, inv_scale):
        matrix_1 = self.key(key).permute(0, 2, 1)
        matrix_2 = self.query(query)
        matrix_3 = matrix_1.matmul(matrix_2)
        matrix_4 = matrix_3 / inv_scale
        matrix_5 = torch.nn.Softmax(dim=-1)(matrix_4)
        out = value.matmul(matrix_5)
        return out

# Initializing the model
m = Model()

# Inputs to the model
N = 8
inv_scale = torch.randn(N) / math.sqrt(16)
query = torch.randn(N, 16)
key = torch.randn(N, 16)
value = torch.randn(N, 16)
