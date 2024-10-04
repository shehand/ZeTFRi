import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from computeModelQuality import convert_model_dict_to_array
from utilsFunctions import calculate_mean_and_stddev, to_var

# Used the original code from https://github.com/danieltan07/dagmm and modified it accordingly

# Helper function for heat level calculation
class CholeskyHelper(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.cholesky(a, False)
        ctx.save_for_backward(l)
        return l
    
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s

# Deep Autoencoding Gaussian Mixture Model (DAGMM)
class DaGMM(nn.Module):
    """Residual Block."""
    def __init__(self, n_gmm = 2, latent_dim=3):
        super(DaGMM, self).__init__()

        layers = []
        # for mnist = 44426, for cifar10(vgg11) = 9756426
        layers += [nn.Linear(9756426,60)]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(60,30)]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(30,10)]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(10,1)]

        self.encoder = nn.Sequential(*layers)


        layers = []
        layers += [nn.Linear(1,10)]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(10,30)]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(30,60)]
        layers += [nn.Tanh()]        
        layers += [nn.Linear(60,9756426)]

        self.decoder = nn.Sequential(*layers)

        layers = []
        layers += [nn.Linear(latent_dim,10)]
        layers += [nn.Tanh()]        
        layers += [nn.Dropout(p=0.5)]        
        layers += [nn.Linear(10,n_gmm)]
        layers += [nn.Softmax(dim=0)]


        self.estimation = nn.Sequential(*layers)

        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm,latent_dim))
        self.register_buffer("cov", torch.zeros(n_gmm,latent_dim,latent_dim))


    def relative_euclidean_distance(self, a, b):
        return (a-b).norm(2, dim=0) / a.norm(2, dim=0)

    def forward(self, xa):
        
        x = to_var(xa[0])
        mean_x, std_x = calculate_mean_and_stddev(xa[0])
        
        enc = self.encoder(x)
        dec = self.decoder(enc)

        recon_error = torch.mean((x - dec) ** 2) # reconstruction error
        model_quality = torch.tensor(xa[1]) # model quality based on the implementation

        rec_cosine = F.cosine_similarity(x, dec, dim=1)
        rec_euclidean = self.relative_euclidean_distance(x, dec)

        # push to cuda if available
        if torch.cuda.is_available():
            model_quality = model_quality.cuda()

        # create the input vector based on the above variables
        z = torch.cat([enc, recon_error.unsqueeze(0), std_x.unsqueeze(0), mean_x.unsqueeze(0), model_quality.unsqueeze(0), rec_euclidean.unsqueeze(0), , rec_cosine.unsqueeze(0)], dim=0) # type: ignore

        gamma = self.estimation(z)

        return enc, dec, z, gamma

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)  # Number of data points

        # Compute sum of gamma along the data points axis
        sum_gamma = torch.sum(gamma, dim=0)

        # Compute phi: K
        phi = sum_gamma / N
        self.phi = phi.data

        # Compute mu: K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data

        # Compute z_mu: N x K x D
        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)

        # Compute z_mu_outer: N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # Compute covariance: K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data

        return phi, mu, cov
        
    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = to_var(self.phi)
        if mu is None:
            mu = to_var(self.mu)
        if cov is None:
            cov = to_var(self.cov)

        k, D = cov.size()

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + to_var(torch.eye(D)*eps)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))

            #det_cov.append(np.linalg.det(cov_k.data.cpu().numpy()* (2*np.pi)))
            det_cov.append((CholeskyHelper.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = torch.cat(det_cov)
        #det_cov = to_var(torch.from_numpy(np.float32(np.array(det_cov))))

        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)

        # move variables to CPU
        max_val = max_val.cpu()
        phi = phi.cpu()
        exp_term = exp_term.cpu()
        det_cov = det_cov.cpu()

        sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim = 1) + eps)
        # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt((2*np.pi)**D * det_cov)).unsqueeze(0), dim = 1) + eps)


        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag


    def loss_function(self, xa, x_hat, z, gamma, lambda_energy, lambda_cov_diag):

        x = to_var(xa[0])
        recon_error = torch.mean((x - x_hat) ** 2)
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)
        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag

        return loss, sample_energy, recon_error, cov_diag