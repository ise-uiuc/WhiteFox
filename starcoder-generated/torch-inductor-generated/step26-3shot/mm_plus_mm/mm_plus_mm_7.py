
class Model(torch.nn.Module):
    def forward(self):
        return torch.ones(5, 5, dtype=torch.float)

model = Model()
# Input to the model
input1 = torch.randn(5, 5)
