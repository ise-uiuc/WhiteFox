
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, m1, m2):
        v1 = torch.matmul(m1, m2.transpose(-2, -1))
        s1 = v1 * self.scale
        s2 = torch.nn.functional.softmax(s1, dim=-1)
        d1 = torch.nn.functional.dropout(s2, p=self.dropout_p)
        v2 = torch.matmul(d1, values)
        return v2

# Initializing the model
m = Model()

# Parameters for the model
m.scale = 100
m.dropout_p = 0.9

# Inputs to the model
m1 = torch.randn(1, 50, 20)
m2 = torch.randn(1, 20, 50)
v = m(m1, m2)

