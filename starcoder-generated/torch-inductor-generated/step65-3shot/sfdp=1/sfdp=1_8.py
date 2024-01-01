
class Model(torch.nn.Module):
    def __init__(self, query_tensor, key_tensor, value_tensor):
        super().__init__()
        self.div = torch.nn.functional.div

        self.matmul_1 = torch.matmul(query_tensor, key_tensor.transpose(-2, -1))
        self.div_1 = self.div([torch.tensor(1.)], [torch.tensor(1.)])

    def forward(self, x1):
        v1 = self.matmul_1(x1)
        v2 = self.div_1(v1, torch.tensor(1.))
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.1)
        v5 = torch.matmul(v4, value_tensor)
        return v5

# Initializing the model
query_tensor = torch.randn(10, 20, 30, 4)
key_tensor = torch.randn(1, 20, 30, 4)
value_tensor = torch.randn(1, 20, 30, 4)
m = Model(query_tensor, key_tensor, value_tensor)

# Input to the model
x1 = torch.randn(10, 20, 30, 4)
