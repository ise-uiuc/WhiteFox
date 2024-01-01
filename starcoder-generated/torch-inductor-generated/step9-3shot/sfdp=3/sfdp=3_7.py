
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.3)
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.permute(0, 1, 3, 2))
        v2 = v1 * 10
        v3 = torch.nn.functional.softmax(v2)
        v4 = self.dropout(v3)
        return torch.matmul(v4, x2)


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 128)
x2 = torch.randn(1, 3, 128, 64)
