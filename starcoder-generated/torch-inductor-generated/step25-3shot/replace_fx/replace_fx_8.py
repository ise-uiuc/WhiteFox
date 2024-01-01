
class MyModule(torch.nn.Module):
    def forward(self, x):
        y0 = x.shape[1]
        shape = (1, x.dim())
        y0 = torch.full(shape, 42., device=x.device)
        x1 = x.detach() # Make a copy of the input tensor, will be detached in the subgraph
        y1 = torch.randn(2, 3, 4)
        y1c = y1.to(dtype=torch.int)
        y2 = torch.rand(2, 3, 4, 7, 5)
        ya = torch.rand((3, 1)) + y2
        yb = ya.tanh().tanh().tanh()
        z1 = y1 / y2
        result1 = torch.mm(z1, y2)
        return result1
# Inputs to the model
x = torch.randn(2,3,4,5,6)
