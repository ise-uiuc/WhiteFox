
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.value = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        k = self.key(x1)
        v = self.value(x1)
        inv_scale = 1.0 / math.sqrt(3 * 8)
        scaled_dot_product = torch.matmul(k, v.transpose(-2, -1)) * inv_scale 
        attention_weights = scaled_dot_product.softmax(dim=-1) 
        output = attention_weights.matmul(v)
        return output
 
# Initializing the model
m = Model()
  
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
