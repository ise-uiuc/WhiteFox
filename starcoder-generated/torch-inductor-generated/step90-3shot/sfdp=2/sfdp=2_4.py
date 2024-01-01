
class Model(torch.nn.Module):
    def __init__(self, query_dim, input_dim, value_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, query_dim)
        self.fc2 = torch.nn.Linear(query_dim, query_dim)
        self.fc3 = torch.nn.Linear(query_dim, value_dim)
 
    def forward(self, x1, x2):
        v1 = x1.matmul(x2.transpose(-2, -1)).div(self.scale_factor).softmax(-1)
        v2 = torch.nn.functional.dropout(v1, p=self.dropout_p)
        v3 = v2.matmul(x2)
        v4 = v1.matmul(x2)
        return v3, v4

# Initializing the model
m = Model(query_dim, input_dim, value_dim)

# Inputs to the model
x1 = torch.randn(10, 5, 42)
x2 = torch.randn(10, 5, 67)
__output_1, __output_2__ = m(x1, x2)

