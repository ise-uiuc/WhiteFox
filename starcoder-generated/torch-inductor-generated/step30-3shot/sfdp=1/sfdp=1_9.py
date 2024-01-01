
class Model(torch.nn.Module):
    def __init__(self, dim=6, input_dim=12, value_dim=8, dropout_p=0.3):
        super().__init__()
        self.dropout_p = dropout_p
        self.dim = dim
        self.query = torch.nn.Linear(input_dim, dim)
        self.key = torch.nn.Linear(input_dim, dim)
        self.value = torch.nn.Linear(input_dim, value_dim)
 
    def forward(self, x1):
        q1 = self.query(x1)
        k1 = self.key(x1)
        v1 = self.value(x1)
        q2 = q1.view(-1, self.dim, 1)
        k2 = k1.view(-1, 1, self.dim)
        q3 = q2.expand(-1, -1, self.dim).flatten(-2, -1)
        k3 = k2.expand(-1, -1, self.dim).flatten(-2, -1)
        q4 = torch.matmul(q3, k3.transpose(-2, -1))
        q5 = q4.div(self.dim ** -0.5)
        q6 = q5.softmax(dim=-1)
        q7 = torch.nn.functional.dropout(q6, p=self.dropout_p)
        q8 = torch.matmul(q7, v1)
        return q8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 12)
