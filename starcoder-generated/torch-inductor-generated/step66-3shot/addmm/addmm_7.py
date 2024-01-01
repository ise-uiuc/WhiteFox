
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lineara = torch.nn.Linear(1261, 512, bias=False)
        self.lineara.weight = torch.nn.Parameter(torch.tensor(np.load('bias_fp16.npy'), dtype=torch.float16))
    def forward(self, x1):
        x1 = self.lineara(x1)
        return x1
# Inputs to the model
x1 = torch.randn(64, 1261, requires_grad=True)
