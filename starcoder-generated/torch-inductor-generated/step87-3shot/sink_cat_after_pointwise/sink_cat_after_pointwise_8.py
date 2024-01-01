
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        # type: (Tensor) -> Tensor
        identity_20 = input.reshape(2, 5, 1, 10)
        identity_30 = identity_20.sum(dim=1)
        identity_40 = input.reshape(1, 1, -1)
        identity_50 = identity_40.view(shape=[1, 1, 10, 6]).sum(dim=2)
        add_10 = torch.add(identity_30, identity_50)
        identity = input + add_10
        return identity
# Inputs to the model
input = torch.randn([1, 5, 4, 6])
