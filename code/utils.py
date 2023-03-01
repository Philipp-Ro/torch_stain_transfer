import yaml

def get_config_from_yaml(config_path):
    with open(file=config_path, mode='r') as param_file:
        parameters = yaml.safe_load(stream=param_file)
    return parameters

def save_config_in_dir(saving_dir,code):
    with open(file=saving_dir, mode='w') as fp:
        yaml.dump(code, fp)

def norm_tensor_to_01(tensor_img):
    tensor_img = tensor_img +1

    # step 2: convert it to [0 ,1]
    tensor_img = tensor_img - tensor_img.min()
    tensor_img_norm = tensor_img / (tensor_img.max() - tensor_img.min())
    return tensor_img_norm 