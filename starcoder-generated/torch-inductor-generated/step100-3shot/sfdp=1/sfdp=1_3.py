
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div_((10000.0))
        v3 = self.softmax(v2)
        v4 = torch.nn.functional.dropout(v3, p = 0.2, training = False)
        v5 = torch.matmul(v4, x2)
        return v5
        
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 128, 128)
x2 = torch.randn(1, 4, 64, 128)
