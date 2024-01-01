
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(2, 64)
    def forward(self, x1):
        t1 = torch.rand_like(self.dense1(x1))
        x2 = self.dense1(x1) * t1
        t2 = F.dropout(x2, p=0.5)
        x3 = torch.nn.functional.relu(t2)
        t3 = torch.nn.functional.dropout(x3, p=0.5)
        t4 = F.dropout(x3, p=0.5)
        t5 = torch.nn.functional.dropout(x3, p=0.5)
        t6 = F.dropout(x4, p=0.5)
        t7 = torch.nn.functional.dropout(x4, p=0.5)
        x5 = t1 + 0.0 * t3 + t4 + t5 + t6 + t7
        return (x2, x5, x3)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
