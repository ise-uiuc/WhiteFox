
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.dropout_p = 0.1
 
    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * 1.0 / math.sqrt(x2.shape[-1])
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p)
        v5 = torch.matmul(v4, x3)
        return v5

# Initializing the model
m = Model()
m = Model().eval()

# Inputs to the model
x1 = torch.randn(1, 100, 80)
x2 = torch.randn(1, 80, 60)
x3 = torch.randn(1, 60, 60)
