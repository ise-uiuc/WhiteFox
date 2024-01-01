
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.dropout = torch.nn.Dropout(0.0)
    def forward(self, data_0):
        t1 = self.dropout(data_0)
        z1 = torch.flatten(t1, start_dim=1, end_dim=-1)
        t2 = self.dropout(data_0)
        y1 = z1 + t2
        return y1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
