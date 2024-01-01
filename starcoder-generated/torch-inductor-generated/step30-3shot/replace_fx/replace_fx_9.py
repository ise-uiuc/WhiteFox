
class TestModel(torch.nn.Module):
    def forward(self, x):
        v = torch.randn([1, 3, 244, 244])
        return v
# Inputs to the model
x = torch.randn([1, 3, 244, 244])
