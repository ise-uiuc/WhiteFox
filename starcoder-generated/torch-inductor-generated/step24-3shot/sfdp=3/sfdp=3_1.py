
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_fc = torch.nn.Dropout(0.1)
 
    def forward(self, x1, x2):
        v1 = self.dropout_fc(x2)
        v2 = self.dropout_fc(x1)
        v3 = torch.matmul(v1, v2.transpose(-2, -1))
        v4 = v3.mul(0.5)
        v5 = torch.nn.functional.softmax(v4, dim=-1)
        v6 = torch.nn.functional.dropout(v5, p=0.1)
        v7 = torch.matmul(v6, x2)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 23)
x2 = torch.randn(3, 50)
