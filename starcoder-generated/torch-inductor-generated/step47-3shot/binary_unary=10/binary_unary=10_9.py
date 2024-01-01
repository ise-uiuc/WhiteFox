
    class Model(torch.nn.Module)
    def forward(self, x1):
        v0 = x1
        v1 = Linear(128, 128).apply(v0)
        v2 = v1 + self.params[1]
        v3 = relu(v2)
        v4 = Linear(128, 1000).apply(x1)
        self.params = [v3]
        return v3

# Initializing the model
    m = Model()
    for p in m.parameters():
        if p.dim() == 2:    # fc weight
            p.data.normal_(0, sqrt(2. / (p.shape[0] + p.shape[1])))
        else:   # conv bias
            n = p.shape[0] * p.shape[1] * p.shape[2]
            p.data.uniform_(-sqrt(6. / n), sqrt(6. / n))

# Inputs to the model
    x1 = th.randn(1, 3, 224, 224)
