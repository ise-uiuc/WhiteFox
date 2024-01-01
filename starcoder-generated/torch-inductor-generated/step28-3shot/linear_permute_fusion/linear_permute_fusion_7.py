
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v122 = self.relu(x1)
        v123 = v122.permute(0, 3, 2, 1)
        v124 = v123.to(dtype=torch.float16, layout=torch.strided, device=torch.device("cuda:0"))
        v125 = v124.to(dtype=torch.float16, layout=torch.strided)
        return v125
# Inputs to the model
x1 = torch.randn(1, 1, 10, 20)
