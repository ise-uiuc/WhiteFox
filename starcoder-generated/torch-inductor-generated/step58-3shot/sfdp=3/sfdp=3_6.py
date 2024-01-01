
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = torch.tensor([0.23])
        self.dropout_p = torch.tensor([0.123])
 
    def forward(self, q1, k1, v1):
        d1 = torch.matmul(q1, k1.transpose(-2, -1))
        d2 = d1 * self.scale_factor
        d3 = torch.nn.functional.dropout(d2.softmax(dim=-1), p=self.dropout_p)
        d4 = d3.matmul(v1)
        return d4

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 3, 128)
k1 = torch.randn(1, 4, 128)
v1 = torch.randn(1, 4, 128)
