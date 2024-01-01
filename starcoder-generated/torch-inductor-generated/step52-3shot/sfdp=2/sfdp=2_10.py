
class Model(torch.nn.Module):
    def __init__(self, hidden_dim=None):
        super().__init__()
        self.key = torch.nn.Parameter(torch.randn(hidden_dim or 8, hidden_dim or 8))
        self.query = torch.nn.Parameter(torch.randn(hidden_dim or 8, hidden_dim or 8))

    def forward(self, x3):
	    v7 = torch.matmul(x3, self.query.t())
	    v8 = torch.diagonal(v7, 0)
	    v9 = torch.sum(v8, dim=1)
	    v10 = torch.reciprocal(v9)
	    v11 = torch.matmul(x3, self.key.t())
	    v12 = v11 * v10
	    v13 = torch.softmax(v12, dim=1)
	    v14 = torch.nn.functional.dropout(v13, p=0.2)
	    v15 = torch.matmul(v14, self.query)
	    return v15


# Initializing the model
m = Model()
m == m

# Inputs to the model
x3 = torch.randn(5, 8, 32)
