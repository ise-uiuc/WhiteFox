
class Model(torch.nn.Module):
    def __init__(self, scale_factor, dropout_p):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p
 
    def forward(self, x1, x2):
        s1 = torch.matmul(x1, x2.transpose(-2, -1))
        s2 = s1.div(self.scale_factor)
        s3 = torch.nn.functional.softmax(s2, dim=-1)
        s4 = torch.nn.functional.dropout(s3, p=self.dropout_p)
        y = torch.matmul(s4, x2)
        return y

# Initializing the model
m = Model(scale_factor=sqrt(1d0), dropout_p=0.5)

# Inputs to the model
x1 = torch.randn(1, 64, 64)
x2 = torch.randn(1, 64, 64)
