
class Model(torch.nn.Module):
    def __init__(self, input_data, weightings):
        super(Model, self).__init__()
        # TODO

    def forward(self, input_data):
        # TODO
        return torch.mm(input_data, self.weightings)
# Inputs to the model
input_data = torch.randn(5, 5)
weightings = torch.randn(5, 5)
