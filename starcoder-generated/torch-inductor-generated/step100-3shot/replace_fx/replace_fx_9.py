
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.2
        self.other_var = torch.ones([])
    def forward(self, x):
        x1 = F.dropout(x, p=self.dropout_p)
        x2 = torch.nn.functional.dropout(x, p=self.dropout_p)
        if self.other_var.item() == 1.0:
            x3 = torch.rand_like(x1)
            x4 = torch.rand_like(x2)
            x5 = torch.rand_like(x1)
        else:
            x6 = torch.rand_like(x1)
        return x
# Inputs to the model
x = torch.randn(1, 2, 2)
