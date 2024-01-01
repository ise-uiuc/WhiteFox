
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x, y):
        b1 = F.dropout(x, p=0.9)
        b2 = torch.rand_like(b1, dtype=torch.float64, layout=torch.strided, device=b1.device, pin_memory=True, requires_grad=False, memory_format=torch.contiguous_format)
        b3 = self.fc(b2).softmax(dim=1)
        return b3, b2
# Inputs to the model
x1 = torch.randn(1, 28, 28)
x2 = 1
