import yaml
import matplotlib.pyplot as plt
import numpy as np 

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

def add_plot(figure, real_HE, fake_IHC ,real_IHC ,num_pairs,p):
    real_HE = real_HE.cpu().detach().numpy()
    fake_IHC = fake_IHC.cpu().detach().numpy()
    real_IHC = real_IHC.cpu().detach().numpy()

    real_HE = np.squeeze(real_HE )
    fake_IHC = np.squeeze(fake_IHC)
    real_IHC = np.squeeze(real_IHC )

    real_HE = np.transpose(real_HE, axes=[1,2,0])
    fake_IHC = np.transpose(fake_IHC, axes=[1,2,0])
    real_IHC = np.transpose(real_IHC, axes=[1,2,0])
    
    if figure == []:
        figure = plt.figure(figsize=(10, 7))
    else:
        rows = num_pairs
        columns = 3

        p=p+1


        figure.add_subplot(rows, columns, p)
        #IHC_batch[n] = (IHC_batch[n] * 127.5 + 127.5).astype(np.uint8)
        plt.imshow(real_HE )
        plt.axis('off')
        plt.title('real_HE')
        p=p+1

        figure.add_subplot(rows, columns, p)
        #HE_batch[n] = (HE_batch[n] * 127.5 + 127.5).astype(np.uint8)
        plt.imshow(fake_IHC )
        plt.axis('off')
        plt.title('fake_IHC')
        p=p+1

        figure.add_subplot(rows, columns, p)
        #fake_HE_batch[n] = (fake_HE_batch[n] * 127.5 + 127.5).astype(np.uint8)
        plt.imshow(real_IHC )
        plt.axis('off')
        plt.title('real_IHC')

    return figure , p