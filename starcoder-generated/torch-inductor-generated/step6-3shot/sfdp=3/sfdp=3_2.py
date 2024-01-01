
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul1 = torch.nn.Linear(512, 64)
 
    def forward(self, x1, x2, scale_factor=1/sqrt(512), dropout_p=0.3):
        v1 = self.matmul1(x1)
        v2 = v1 * scale_factor
        v3 = torch.nn.functional.dropout(v2, p=dropout_p)
        v4 = v3.matmul(x2)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512)
x2 = torch.randn(1, 64, 64)
