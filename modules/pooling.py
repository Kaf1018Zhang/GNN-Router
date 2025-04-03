from torch_geometric.nn import global_mean_pool, global_max_pool, TopKPooling


def global_mean_pooling(x, batch):
    return global_mean_pool(x, batch)


def global_max_pooling(x, batch):
    return global_max_pool(x, batch)


class TopKPoolWrapper:
    def __init__(self, ratio=0.8):
        self.pool = TopKPooling(in_channels=None, ratio=ratio)  # in_channels will be set later

    def __call__(self, x, edge_index, batch):
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
        return x, batch
