
class Model(torch.nn.Module):
    def __init__(self, dim, dropout_p=0.1):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.linear2 = torch.nn.Linear(dim, dim)
 
    def forward(self, q1, k1, v1):
        q2 = self.linear1(q1)
        q3 = self.dropout(q2)
        q4 = self.linear2(q3)
        k2 = self.linear1(k1)
        k3 = self.dropout(k2)
        k4 = self.linear2(k3)
        v2 = self.linear1(v1)
        v3 = self.dropout(v2)
        v4 = self.linear2(v3)
        q5 = torch.matmul(q4, k4.transpose(-2, -1))
        v5 = torch.matmul(v4, k4.transpose(-2, -1))
        scale_factor = q5.size(-1) ** -0.5
        q6 = torch.matmul(q5, k5) * scale_factor
        q7 = self.dropout(q6)
        output = torch.matmul(q7, v5)
        return output

# Initializing the model
dim = 16
m = Model(dim)

# Inputs to the model
q1 = torch.randn(3, dim, requires_grad=True)
k1 = torch.randn(3, dim, requires_grad=True)
v1 = torch.randn(3, dim, requires_grad=True)
