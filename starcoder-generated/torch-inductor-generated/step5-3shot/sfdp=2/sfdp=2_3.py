
class Model(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.dropout_ratio = 0.5
      self.dropout_p = 0.05
      self.scale_factor = math.sqrt(self.dropout_ratio)
    
    def forward(self, x1):
        v1 = torch.matmul(x1, x1.transpose(-2, -1))
        v2 = v1.div(self.scale_factor)
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p)
        v5 = torch.matmul(v4, x1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 48, 256)
