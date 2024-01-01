
class Model(torch.nn.Module):
    def forward(self, input_tensor):
        q = torch.randn(1, 32, 64)
        k = torch.randn(1, 32, 64)
        v = torch.randn(1, 32, 64)
        out = torch.nn.functional.softmax(torch.matmul(q, k.transpose(-2, -1)) * 0.5, dim=-1)
        out = torch.nn.functional.dropout(out, p=.2)
        return torch.matmul(out, v)

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(1, 32, 64)
