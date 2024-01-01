
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 8)
        self.linear2 = torch.nn.Linear(8, 32)
        self.linear3 = torch.nn.Linear(32, 64)
        self.linear4 = torch.nn.Linear(64, 64)
        self.linear5 = torch.nn.Linear(64, 32)
        self.linear6 = torch.nn.Linear(32, 8)
        self.linear7 = torch.nn.Linear(8, 64)
        self.linear8 = torch.nn.Linear(64, 32)
        self.linear9 = torch.nn.Linear(32, 1)
    def forward(self, x1):
        v1 = (x1).permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        v3 = torch.nn.functional.linear(v2, self.linear2.weight, self.linear2.bias)
        v4 = self.linear3(v3)
        v3 = torch.nn.functional.relu(v4)
        v2 = torch.nn.functional.linear(v3, self.linear4.weight, self.linear4.bias)
        v2 = v2.unsqueeze(-1).unsqueeze(-1)
        v3 = torch.nn.functional.linear(v2, self.linear5.weight, self.linear5.bias)
        v3 = v3.reshape(1, 32, 32).to(dtype=torch.float16)
        v4 = (v3 - 1.37) - 2.74
        v4 = torch.nn.functional.linear(v4, self.linear6.weight, self.linear6.bias)
        v4 = torch.nn.functional.relu(v4)
        v5 = self.linear7(v2)
        v5 = torch.nn.functional.linear(v5, self.linear8.weight, self.linear8.bias)
        v5 = torch.nn.functional.linear(v4, self.linear9.weight, self.linear9.bias)
        v1 = (2.37).unsqueeze(dim=0).permute(0, 2, 1)
        v6 = v1 + v3.permute(1, 0, 2)
        v1 = v6 + 2.74
        x2 = (v4 + 3.07)
        v2 = v2 + v5
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
