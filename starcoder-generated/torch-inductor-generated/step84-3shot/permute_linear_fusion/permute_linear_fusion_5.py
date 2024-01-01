
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.unsqueeze(dim=-1)
        v3 = v3 * v1.unsqueeze(dim=1)
        v3 = torch.matmul(v3, v1.unsqueeze(dim=1)).squeeze(dim=-1).t()
        v3 = self.sigmoid(v3)
        v3 = v3.unsqueeze(dim=-1) + v2.unsqueeze(dim=-1)
        v3 = v3.unsqueeze(dim=-1) # TODO: What's this line used for?
        # TODO: Please figure out v3's shape and the shape of elements in this tensor in the computational graph.
        return v3
# Inputs to the model
x1 = torch.randn(2,2,2)
