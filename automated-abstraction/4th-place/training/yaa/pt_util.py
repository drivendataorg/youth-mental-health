def requires_grad(module, requires_grad):
    for layer in module.children():
        for param in layer.parameters():
            param.requires_grad = requires_grad


