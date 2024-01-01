
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_scale = math.sqrt(query.shape[0])
        self.query = torch.nn.init.normal_(torch.randn(1, 512, 100, 40), -self.inv_scale, self.inv_scale)
        self.key = torch.nn.init.normal_(torch.randn(1, 512, 200, 160), -self.inv_scale, self.inv_scale)
        self.value = torch.nn.init.normal_(torch.randn(1, 1024, 200, 160), -1, 1)
 
    def forward(self):
        v = torch.matmul(self.input @ self.query, self.key.transpose(-2, -1))
        v = v / self.inv_scale
        v = v.softmax(dim=-1).matmul(self.value)
        return v
   
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512, 100, 40)
x2 = torch.randn(1, 512, 200, 160)
x3 = torch.randn(1, 1024, 200, 160)
