
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.5

    def forward(self, m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor, m4: torch.Tensor):
        v1 = torch.matmul(m1, m2.transpose(-2, -1))
        v2 = v1 * (1 / 0.1)
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p)
        v5 = torch.matmul(v4, m3)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
m1 = torch.randn(1, 3, 2, 4)
m2 = torch.randn(1, 4, 5, 6)
m3 = torch.randn(1, 6, 9)
m4 = torch.randn(1, 2, 5)
