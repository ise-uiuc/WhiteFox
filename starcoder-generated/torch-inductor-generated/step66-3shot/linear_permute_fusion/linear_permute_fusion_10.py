
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2048, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256 + 96, 1)
    def forward(self, x):
        v1 = torch.nn.functional.linear(x, self.linear1.weight, self.linear1.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v2, self.linear2.weight, self.linear2.bias)
        v4 = v3.view(-1, v3.size(0), v3.size(1) * v3.size(2))
        v5 = v2[:, :, 53:96]
        v6 = v2.index_select(2, torch.tensor([52, 95]).to(torch.int64), )
        v8 = torch.cat((v5, v6), 2)
        v7 = torch.nn.functional.linear(v8, self.linear3.weight, self.linear3.bias)
        return v7
# Inputs to the model
x = torch.randn(5, 2048, 16)
