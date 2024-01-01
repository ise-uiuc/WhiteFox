
class Model(torch.nn.Module):
    def forward(self, x):
        v1 = x.detach().requires_grad_()
        v2 = v1 + x
        return torch.autograd.grad(v2, v1, create_graph=True)
# Inputs to the model
x = torch.tensor([[2.0], [0.1]])
