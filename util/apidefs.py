def _get_api_defs(api_def_fn):
    """Get argument name list from api definition.

    Example:
        tf.math.accumulate_n(inputs,shape=None,tensor_dtype=None,name=None) ->
        {
            'args': ['inputs', 'shape', 'tensor_dtype']
        }
        tf.linalg.einsum(equation,*inputs,**kwargs) ->
        {
            'args': ['equation']
            'def': 'tf.linalg.einsum(equation,*inputs,**kwargs)'
        }
    #TODO: the follow information can also guide our argument/keyword mutation.
     - the optional arguments `shape=None` -> shape=<SPAN>
     - *inputs, -> <SPAN>,
     - **kwargs  -> <SPAN>=<SPAN>

    """
    with open(api_def_fn, "r") as f:
        data = f.read().splitlines()

    api_defs = dict()
    for line in data:
        api, argstr = line.split("(", 1)

        args = []
        argstr = argstr.split(",")

        for a in argstr:
            a = a.strip()
            var = a if "=" not in a else a.split("=", 1)[0]
            if var.isidentifier() and var != "name":
                args.append(var)

        api_defs[api] = {"args": args, "def": line}

    return api_defs


def get_api_defs(lib):
    assert lib in ["tf", "torch"]
    if lib == "tf":
        api_def_fn = "data/api_def_tf.txt"
    else:
        api_def_fn = "data/api_def_torch.txt"
    return _get_api_defs(api_def_fn)


def _test_apidefs():
    tf_api_defs = get_api_defs("tf")
    assert tf_api_defs["tf.math.accumulate_n"]["args"] == [
        "inputs",
        "shape",
        "tensor_dtype",
    ]
    assert tf_api_defs["tf.linalg.einsum"]["args"] == ["equation"]

    torch_api_defs = get_api_defs("torch")
    # torch.Tensor.divide(_input_tensor, value, *, rounding_mode=None)
    assert torch_api_defs["torch.Tensor.divide"]["args"] == [
        "_input_tensor",
        "value",
        "rounding_mode",
    ]


if __name__ == "__main__":
    _test_apidefs()
