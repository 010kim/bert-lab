import re

def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
    """Checks whether the casing config is consistent with the checkpoint name"""

    if not init_checkpoint:
        return
    
    m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
    if m is None:
        return
    
    model_name = m.group(1)
    lower_models = ["uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12", "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"]
    cased_models = ["cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16", "multi_cased_L-12_H-768_A-12"]

    is_bad_config = False
    if model_name in lower_models and not do_lower_case:
        is_bad_config = True
        actual_flag = "False"
        case_name = "lowercased"
        opposite_flag = "True"

    if model_name in cased_models and do_lower_case:
        is_bad_config = True
        actual_flag = "True"
        case_name = "cased"
        opposite_flag = "False"

    if is_bad_config:
        raise ValueError("do_lower_case and model do not match")
    
