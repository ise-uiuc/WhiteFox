
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.9
        self.inv_scale_factor = 1.0 / np.sqrt(32)
 
    def forward(self, x1, x2, x3):
        q = torch.matmul(x1, x2.transpose(-2, -1) / self.inv_scale_factor)
        d = torch.nn.functional.dropout(q.softmax(dim=-1), p=self.dropout_p)
        output = torch.matmul(d, x3)
        return output
        
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 64)
x2 = torch.randn(1, 64, 32)
x3 = torch.randn(1, 64, 32)
