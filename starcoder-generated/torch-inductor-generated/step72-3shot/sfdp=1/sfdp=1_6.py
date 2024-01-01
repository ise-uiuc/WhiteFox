
class Model(torch.nn.Module):
    def __init__(self, input_dim=128, mlp_dim=128):
        super().__init__()
        self.dropout_p = 0.1
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mat1 = torch.nn.Linear(input_dim, mlp_dim)
        self.mat2 = torch.nn.Linear(mlp_dim, input_dim)
        self.scale_factor = 32.

    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(self.scale_factor)
        v3 = self.softmax(v2)
        v4 = self.dropout(v3)
        v5 = self.mat1(v4)
        v6 = self.mat2(v5)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
x2 = torch.randn(1, 128)
