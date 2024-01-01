
class Model(torch.nn.Module):
    def init(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.layer_1 = torch.nn.Linear(self.input_dim, self.input_dim)
        self.layer_2 = torch.nn.Linear(self.input_dim, self.input_dim)
    def forward(x)
        v1 = self.layer_1(x.transpose(-2, -1).flatten(-2)).reshape(x.shape[0], -1, self.input_dim).permute(...) # Incomplete pattern
        return torch.nn.functional.linear(v1,...)
# Inputs to the model
x = torch.randn(10, 10, 10)
