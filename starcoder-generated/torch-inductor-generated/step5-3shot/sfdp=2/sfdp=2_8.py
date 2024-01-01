
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_scale_factor = np.random.uniform(10.0, 20.0)
 
    def forward(self, x0, x1):
        v0 = torch.matmul(x0, x1.transpose(-2, -1)) * self.inv_scale_factor
        v1 = torch.nn.functional.softmax(v0, dim=-1)
        v2 = self.dropout(v1, p=0.1)
        v3 = torch.matmul(v2, x1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 10, 128)
x1 = torch.randn(1, 128, 100)
