
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q2, k3, v4, i5):
        s1 = torch.matmul(q2, k3.transpose(-2, -1))
        s2 = s1 / i5
        s3 = torch.nn.functional.softmax(s2, dim=-1)
        s4 = torch.nn.functional.dropout(s3, i5, training=True)
        return torch.matmul(s4, v4)

# Initializing the model
m = Model()

# Inputs to the model
q2 = torch.randn(8, 25, 768)
k3 = torch.randn(8, 15, 768)
v4 = torch.randn(8, 15, 768)
i5 = torch.randint(4, (1,))
