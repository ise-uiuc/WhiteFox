
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.dropout_p = torch.tensor([0.5])
        
    def forward(self, q, k, v):
        m1 = torch.matmul(q, k.transpose(-2, -1))
        s1 = m1.div(self.scale_factor)
        s2 = torch.nn.functional.dropout(s1, p=self.dropout_p)
        o = s2.matmul(v)
        return o

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 8, 128, 128)
k = torch.randn(1, 8, 128, 128)
v = torch.randn(1, 8, 128, 128)
