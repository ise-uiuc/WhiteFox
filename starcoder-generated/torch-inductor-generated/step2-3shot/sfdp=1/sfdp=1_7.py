
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.5
        self.inv_scale_factor = math.sqrt(1 / 128)
 
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(self.inv_scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p)
        v5 = v4.matmul(x3)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3, 256)
x2 = torch.randn(2, 3, 128)
x3 = torch.randn(2, 128, 64)
