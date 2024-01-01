
class Model(torch.nn.Module):
    def __init__(self, inv_scale_factor):
        super().__init__()
        self.query = torch.nn.Linear(8, 8)
        self.key = torch.nn.Linear(8, 8)
        self.value = torch.nn.Linear(8, 8)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.inv_scale_factor = inv_scale_factor
 
    def forward(self, x, y):
        q1 = self.query(x)
        k1 = self.key(y)
        v1 = self.value(y)
        q2 = q1.transpose(-2, -1)
        k2 = k1.transpose(-2, -1)
        q3 = q2.div(self.inv_scale_factor)
        q4 = q3.softmax(dim=-1)
        d1 = self.dropout(q4)
        o1 = d1.matmul(v1)
        return o1

# Initializing the model
m = Model(inv_scale_factor=float(1 / (np.sqrt(8) * np.sqrt(8))))

# Inputs to the model
x = torch.randn(1, 8)
y = torch.randn(2, 8)
