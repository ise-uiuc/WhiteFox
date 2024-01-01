
class Model(torch.nn.Module):
    def __init__(self, hidden, in_features, out_features, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, x, weight):
        v1 = torch.matmul(x, weight.transpose(-2, -1))
        v2 = v1.div(self.in_features**.5)
        v3 = torch.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, self.dropout_p, training=self.training)
        v5 = torch.matmul(v4, weight)
        return v5

# Initializing the model
m = Model(32, 64, 100, 0.1)

# Inputs to the model
x = torch.randn(28, 64)
weight = torch.randn(100, 64)
