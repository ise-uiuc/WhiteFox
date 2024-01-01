
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, x1, x2):
        x3 = query.matmul(key.transpose(-2, -1))
        x4 = x3 / (1e-6 + x3.flatten(2).max(-1).values.unsqueeze(-1).unsqueeze(-1))
        x5 = torch.nn.functional.softmax(x4, dim=-1)
        x6 = torch.nn.functional.dropout(x5, p=self.dropout_p)
        return x6.matmul(value)

# Initializing the model
m = Model(0.1)

# Inputs to the model
x1 = torch.randn(2, 3, 4, 5)
x2 = torch.randn(2, 3, 5, 2)
