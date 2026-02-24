from .vit_classifier import ViTClassifier


def get_target_modules(
    vit_classifier: ViTClassifier, partial=None, no_tuning: bool = False
):
    n_layers = len(vit_classifier.vit_model.encoder.layer)

    block_module_name = "vit_model.encoder.layer"
    get_attn_in_module_names = lambda i: [
        f"{block_module_name}.{i}.attention.attention.query",
        # f"{block_module_name}.{i}.attention.attention.key",
        f"{block_module_name}.{i}.attention.attention.value",
    ]
    # get_attn_out_module_names = lambda i: f"{block_module_name}.{i}.output.dense"
    # get_mlp_in_module_names = lambda i: f"{block_module_name}.{i}.mlp.fc1"
    # get_mlp_out_module_names = lambda i: f"{block_module_name}.{i}.mlp.fc2"
    # get_ln_modules_names = lambda i: f"vit_model.layernorm"

    if partial is None:
        _start, _end = 0, n_layers
    elif isinstance(partial, int):
        _start, _end = n_layers - partial, n_layers
    elif isinstance(partial, list):
        _start, _end = partial[0], partial[1]

    target_modules = []
    for i in range(_start, _end):
        target_modules.extend([*get_attn_in_module_names(i)])

    return target_modules
