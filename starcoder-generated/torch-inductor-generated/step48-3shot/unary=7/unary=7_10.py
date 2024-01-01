
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1[0] if 1 == len(v1.size()) and 1 == v1.size()[0] else v1
        v3 = v2 * 0.3333332981853485 + 0.16667
        v4 = v2 * 0.5 + 0.25
        v5 = v2 * 0.6666667461395264 + 0.5
        v6 = max(max(max(v3, v4), v5), 0)
        v7 = v6 / 6
        return v7

# Initializing the model
m = Model()
(m.linear.weight, m.linear.bias) = (
    torch.tensor([[0.32865377, 0.35843190, 0.55123995, 0.37618194, 0.58888664, 0.66060278, 0.85601045, 0.10255317]]),
    torch.tensor([[-0.82540384]]),
)

# Inputs to the model
x1 = torch.randn(1, 8)
