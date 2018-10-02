import protonets.data

def load(opt, splits):
    if opt['data.dataset'] == 'omniglot':
        if opt['data.encoder'] == 'acai':
            ds = protonets.data.omniglot_cache.load(opt, splits)
        elif opt['data.encoder'] == 'none':
            ds = protonets.data.omniglot.load(opt, splits)
    else:
        raise ValueError("Unknown dataset: {:s}".format(opt['data.dataset']))

    return ds
