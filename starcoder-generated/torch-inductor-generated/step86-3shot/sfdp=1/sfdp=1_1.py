
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.5, scale_factor=2**-0.5):
        super().__init__()
        self.dropout_p = dropout_p
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.scale_factor = scale_factor
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1,x2.transpose(-2, -1))
        v2 = v1.div(self.scale_factor)
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = self.dropout(v3)
        v5 = torch.matmul(v4, x2)
        return v5

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 30, 512)
x2 = torch.randn(1, 512, 896)
