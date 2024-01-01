
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m_list = torch.nn.ModuleList([torch.nn.Conv2d(3, 3, 3) for _ in range(6)])
    def forward(self, x2):
        for i in range(len(self.m_list)):
            x2 = self.m_list[i](x2)
        return x2
# Inputs to the model
x2 = torch.randn(1, 3, 4, 4)
