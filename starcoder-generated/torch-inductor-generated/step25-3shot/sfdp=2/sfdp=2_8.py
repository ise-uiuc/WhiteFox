
class Model(torch.nn.Module):
    def __init__(self, num_heads=2, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.scale_factor = 1 / (dropout * num_heads)
 
    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(self.scale_factor)
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout)
        output = v4.matmul(x3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 5)
x2 = torch.randn(1, 2, 4)
x3 = torch.randn(1, 4, 6)
