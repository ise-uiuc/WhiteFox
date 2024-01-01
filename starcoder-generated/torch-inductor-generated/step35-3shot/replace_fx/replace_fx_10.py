
class Model(torch.nn.Module):
    def forward(self, x, y):
        temp = torch.rand_like(x)
        x = torch.nn.functional.dropout(temp, training=True)
        if x.device == "cuda":
            res = 2*x
        else:
            res = 3*x
        y = torch.where(res > 1, torch.ones_like(res), torch.zeros_like(res))
        return y
# Inputs to the model
x1 = torch.rand((1, 2, 2), dtype=torch.float)
x2 = torch.rand((1, 2, 2), dtype=torch.float)
