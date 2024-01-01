
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, q1, k1):
        s1 = torch.matmul(q1, k1.transpose(-2, -1))
        s2 = s1 * 0.5
        s3 = s2.transpose(-2, -1)
        s4 = self.softmax(s3)
        s5 = torch.nn.functional.dropout(s4, p=0.5)
        v1 = torch.matmul(s5, v1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(8, 256, 64, 64)
k1 = torch.randn(8, 256, 64, 64)
