
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
     
        self.dropout = torch.nn.functional.dropout
 
    def forward(self, q1, k1, v1, dropout_p):
        q2 = self.dropout(q1, p=dropout_p)
        k2 = self.dropout(k1, p=dropout_p)
        v2 = self.dropout(v1, p=dropout_p)
 
        q3 = torch.matmul(q2, k2.transpose(-2, -1))
        k3 = q3.div(inv_scale_factor)
        v3 = k3.softmax(dim=-1)
 
        output = v3.matmul(v2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 120, 64)
k1 = torch.randn(1, 120, 64)
v1 = torch.randn(1, 120, 64)
dropout_p = 0.05
