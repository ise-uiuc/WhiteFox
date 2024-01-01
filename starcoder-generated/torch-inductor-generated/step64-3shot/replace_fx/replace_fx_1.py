
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.conv = torch.nn.Conv2d(2, 2, 2)
    def forward(self, x0_1):
        x1_1 = self.conv(x0_1)
        x2_1 = self.linear(x1_1)
        x3_1 = torch.rand_like(x1_1)
        x4_1 = torch.nn.functional.dropout(x3_1)
        x5_1 = torch.nn.functional.dropout(x4_1)
        return x5_1
# Inputs to the model
x0_1 = torch.rand(2, 2, 2, 2)
