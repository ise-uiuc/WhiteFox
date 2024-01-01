
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        result = input
        for _ in range(10):
            result = torch.nn.functional.dropout(result, p=0.5)
        return result
x = torch.randn(10, 10)
