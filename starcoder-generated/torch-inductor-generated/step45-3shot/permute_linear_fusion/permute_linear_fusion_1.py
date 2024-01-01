
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)
    def forward(self, x1):
        n = torch.numel(self.layer.weight)
        m3 = torch.zeros(n, device=self.layer.weight.device)
        m4 = torch.tensor(0, device=self.layer.weight.device)
        i = 0
        j = 0
        while i < len(m3):
            m3[i] += self.layer.weight[i // 2, i % 2]
            m4 += self.layer.weight[i // 2, i % 2]
            i += 1
        return torch.cat([x1, self.layer.weight.unsqueeze(0)], dim=0) + n - m4
# Inputs to the model
x1 = torch.randn(2, 1, 2)
