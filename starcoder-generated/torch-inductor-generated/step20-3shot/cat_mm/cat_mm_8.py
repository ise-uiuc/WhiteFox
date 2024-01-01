
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(15, 1)
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        out = []
        v2 = torch.mm(x1, x2)
        for i in range(5):
            out.append(torch.cat(((v1.unsqueeze(0)), v2.unsqueeze(0)), dim=1))
            v2 = torch.mm(x1, v2)
        result = torch.stack(out)
        result = result.view(5, 2, 10)
        result = self.fc(result)
        return torch.cat([result, v2], dim=1)
# Inputs to the model
x1 = torch.randn(4, 3)
x2 = torch.randn(3, 5)
