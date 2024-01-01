
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(32, 32, 64))
        self.key = torch.nn.Parameter(torch.randn(32, 32, 64))
        self.value = torch.nn.Parameter(torch.randn(32, 32, 64))
        self.atten_mask = torch.randn(1, 1, 1, 32)
 
    def forward(self, x1):
        kq = self.query @ self.key.transpose(-2, -1) # Compute the key @ value matrix
        kq = kq / math.sqrt(self.query.size(-1))
        kq = kq + self.atten_mask
        atten_w = torch.softmax(kq, dim=-1) # Apply softmax to the scaled dot product
        output = atten_w @ self.value # Compute the output
        return output

# Initializing the model
x1 = torch.randn(1, 1, 32)
x2 = torch.randn(1, 1, 32)
x3 = torch.randn(1, 1, 32)
m = Model()
