
class Model(torch.nn.Module):
    def forward(self, data_0):
        t1 = torch.nn.functional.dropout(data_0, training = False)
        t2 = torch.rand_like(data_0, dtype=torch.float)
        z1 = t2 + t1
        t3 = torch.nn.functional.dropout(data_0)
        y1 = z1 + t3
        return y1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
