
# TODO: How do we generate inputs?
# TODO: Why should other_tensor + x1 generate a model?
class MyModule(torch.nn.Module):
    def __init__(self):
        # TODO: Fill in missing information
        super().__init__()
        # TODO: Fill in missing information
        self.conv = torch.nn.Conv2d(28, 521, 1, stride=1, padding=1)
    def forward(self, x3, x2, other_tensor):
        # TODO: Apply your model
        return x2
