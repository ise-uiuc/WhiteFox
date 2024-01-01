
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
        self.dropout_p = torch.nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
 
    def forward(self, x1, x2, x3, x4):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * self.scale_factor
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p)
        v5 = torch.matmul(v4, x3)
        return v5
    
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128, 64)
x2 = torch.randn(1, 128, 64)
x3 = torch.randn(1, 64, 256)
x4 = torch.randn(1, 128, 256)
