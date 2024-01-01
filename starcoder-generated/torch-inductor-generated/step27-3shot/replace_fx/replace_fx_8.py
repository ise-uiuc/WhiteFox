
class Model(torch.nn.Module):
    def __init__(self, p1):
        super().__init__()
        self.a1 = torch.nn.functional.dropout(p1)
        self.a2 = torch.nn.functional.dropout(p1)
    def forward(self, x):
        x = self.a1(x) + self.a2(x)
        x = torch.nn.functional.dropout(x, p=0.8, training=False)
        x = torch.nn.functional.dropout(x, p=0.9, training=False)
        x = torch.rand_like(x)
        return 1
p1 = torch.randn(1, requires_grad=True)
# Inputs to the model
print(p1)
# print(x1)
