class GraphAttention(AsyncModule):
    def __init__(self,
        in_features: int,
        out_features: int,
        num_layers: int = 1,
        bias: bool = True,
        shortcut: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_layers = int(num_layers)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = self.in_features if i == 0 else self.out_features
            out_ch = self.out_features
            self.convs.append(GCNConv(in_ch,out_ch, bias=bias, shortcut=shortcut))
    
    def reset_parameters(self) -> None:
        for conv in self.convs:
            cast(GCNConv, conv).reset_parameters()

    async def async_forward(self, g: DGLGraph, x: Tensor | None = None) -> Tensor:
        x, route, route_first = GCNConv.get_inputs(g, x)

        for i, conv in enumerate(self.convs):
            conv = cast(GCNConv, conv)
            if i == 0:
                r = route if route_first else None
                x = await conv.async_forward(g, x, route=r)
            else:
                x = F.relu(x)
                r = route
                x = await conv.async_forward(g, x, route=r)
        return x
