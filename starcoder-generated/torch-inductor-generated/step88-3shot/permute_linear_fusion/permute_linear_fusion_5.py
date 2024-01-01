
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.softmax = torch.nn.Identity()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.squeeze(-1)
        v4 = self.softmax(v3)
        v5 = v4.unsqueeze(1)
        v6 = v5.transpose(1,2)
        v7 = torch.sum(v6,dim=2, keepdim=True)
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 2)
