
class Model(torch.nn.Module):
    def __init__():
        super().__init__()
    def forward(self, x, *other):
        results = []
        for each in other:
            result = torch.mm(x, each)
            results.append(result)
        v = results[0]
        for res in results:
            v = v.add(res)
        return v
# Inputs to the model
x = torch.randn(5, 4)
other = []
for i in range(4):
    other.append(torch.randn(5, 4))
