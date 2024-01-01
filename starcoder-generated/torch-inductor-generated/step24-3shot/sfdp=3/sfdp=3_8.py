
class Model(torch.nn.Module):
    def __init__(self, scale_factor, dropout_p):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p
 
    def forward(self, x, y):
        v1 = torch.matmul(x, y.transpose(-2, -1))
        v2 = v1 * self.scale_factor
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p)
        v5 = torch.matmul(v4, y)
        return v5

# Initializing the model
m = Model(scale_factor=1.0/np.sqrt(2), dropout_p=0.1)

# Inputs to the model
x = torch.randn(1, 32, 3, 48, 3)
y = torch.randn(1, 64, 3, 24, 12)
