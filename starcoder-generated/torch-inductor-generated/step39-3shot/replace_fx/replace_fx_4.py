
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = torch.nn.functional.dropout(torch.randn(5, 5), p=0.8)
        self.t_2 = torch.nn.functional.dropout(self.t, p=0.8)
        self.t_3 = self.t_2
    def forward(self, x):
        return self.t + self.t_3
# Inputs to the model
x1 = torch.randn(1, 2)
