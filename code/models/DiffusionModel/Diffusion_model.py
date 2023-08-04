import torch
import tqdm


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=[256,256] ,device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size[0]
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_img(self, x, t):
        # -----------------------------------------------------------------------------------------------------------------
        # the noise doesnt have to be calculated itterativewly

        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    

    def get_index_from_list(vals, t, x_shape):
        # --------------------------------------------------------------
        # Returns a specific index t of a passed list of values vals
        # while considering the batch dimension.
        # --------------------------------------------------------------

        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(x_0, t, device="cuda"):
        # --------------------------------------------------------------
        # Takes an image and a timestep as input and 
        # returns the noisy version of it
        # --------------------------------------------------------------
        print(x_0)
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

    def sample(self, model, n):
        #logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        
        return x


#-----------------------------------------------------------------------------------------------
# POSITIONAL EMBEDDING
#-----------------------------------------------------------------------------------------------
def get_positional_Embeddings(sequence_length, embedding_dim):
    # the n variable is scalling the values in the positional embedding in the attention is all you need paper n=10000 was choosen 
    n = 10000
    result = torch.ones(sequence_length, embedding_dim)
    for i in range(sequence_length):
        for j in range(embedding_dim):
            result[i][j] = np.sin(i / (n ** (j / embedding_dim))) if j % 2 == 0 else np.cos(i / (n ** ((j - 1) / embedding_dim)))
    return result
