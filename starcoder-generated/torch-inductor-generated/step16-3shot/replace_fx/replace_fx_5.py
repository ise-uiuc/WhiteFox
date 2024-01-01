
class model7(torch.nn.Module):
    def forward(self, x):
        x = torch.rand_like(x)
        return torch.nn.functional.dropout(x)
# Inputs to the model
x = torch.rand(2, 2)
