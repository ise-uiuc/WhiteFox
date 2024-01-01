
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.randint(low=-3, high=4, size=(2, 20, 200))
        self.key = torch.randint(low=-3, high=4, size=(10, 20, 100))
        self.value = torch.randint(low=-3, high=4, size=(10, 20, 150))
 
    def forward(self, query, key, value):
        v0 = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = pow(query.shape[-1], -0.25)
        v1 = v0.div(inv_scale_factor)
        v2 = v1.softmax(dim=-1)
        v3 = torch.nn.functional.dropout(v2, p=0.6)
        output = v3.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 20, 200)
key = torch.randn(10, 20, 100)
value = torch.randn(10, 20, 150)
