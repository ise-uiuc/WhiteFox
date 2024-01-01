
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = torch.tensor(0.33, dtype=torch.float32)
 
 
    def forward(self, x1, x2):
        x1_dot_x2 = torch.matmul(x1, x2.transpose(-2, -1))
        x3 = x1_dot_x2.mul(self.scale_factor)
        x4 = x3.softmax(dim=-1)
        x5 = torch.nn.functional.dropout(x4, p=0.1)
        return torch.matmul(x5, x2)
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 128)
x2 = torch.randn(1, 32, 128)
