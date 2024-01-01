
class Model(torch.nn.Module):
    def __init__(self, dropout_p, scale_factor):
        super().__init__()
        self.dropout_p = dropout_p
        self.scale_factor = scale_factor
 
    def forward(self, x1, x2, x3):
        q1 = torch.matmul(x1, x2.transpose(-2, -1))
        q2 = q1.div(self.scale_factor)
        q3 = q2.softmax(dim=-1)
        q4 = torch.nn.functional.dropout(q3, p=self.dropout_p)
        q5 = torch.matmul(q4, x3)
        return q5

# Initializing the model
m = Model(dropout_p=0.05, scale_factor=32.0)

# Inputs to the model
x1 = torch.randn(1, 64, 32, 32)
x2 = torch.randn(1, 64, 32, 32)
x3 = torch.randn(1, 64, 32, 32)
