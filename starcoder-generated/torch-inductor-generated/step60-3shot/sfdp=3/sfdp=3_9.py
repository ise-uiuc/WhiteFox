
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.0
        
        self.scale_factor = 100.0
        self.dropout = torch.nn.Dropout(p=self.dropout_p)
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, x2, x3):
        v1 = torch.matmul(x2, x3.transpose(-2, -1))
        v2 = v1 * self.scale_factor
        v3 = self.softmax(v2)
        v4 = self.dropout(v3)
        output = v4.matmul(x3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(6, 3, 16)
x3 = torch.randn(6, 3, 256)
