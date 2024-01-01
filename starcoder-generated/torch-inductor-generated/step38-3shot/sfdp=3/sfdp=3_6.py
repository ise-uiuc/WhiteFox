
class Model(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.W_query = torch.nn.Parameter(torch.Tensor(n, n))
        self.W_key = torch.nn.Parameter(torch.Tensor(n, n))
        self.W_value = torch.nn.Parameter(torch.Tensor(n, n))
        self.scale_factor = math.sqrt(n)
 
    def forward(self, x):
        qy = torch.matmul(x, self.W_query)
        ky = torch.matmul(x, self.W_key)
        vy = torch.matmul(x, self.W_value)
        s1 = qy.mul(self.scale_factor)
        s2 = s1.softmax(dim=-1)
        s3 = torch.nn.functional.dropout(s2, p=0.05)
        z = s3.matmul(vy)
        return z

# Initializing the model
model = Model(3)

# Inputs to the model
batch = torch.randn((10, 3, 7, 7))
