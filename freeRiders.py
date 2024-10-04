from copy import deepcopy
import torch
from computeModelQuality import compute_model_quality_with_loss_diff, convert_model_dict_to_array
from DAGMMModel import DaGMM
from tqdm import tqdm
import statistics as stat

class FreeRiderDetection:
    def __init__(self, fr_method):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.fr_method = fr_method
        self.gmm_k = 4
        self.lr = 1e-4
        self.num_epochs = 200
        self.lambda_energy = 0.1
        self.lambda_cov_diag = 0.005

        self.heat_values = []

        self.dagmm = DaGMM(7, 7)

        self.optimizer = torch.optim.Adam(self.dagmm.parameters(), lr=self.lr)

    def detect_free_riders(self, local_models, loss_diffs, data_dist): # loss_dff means the loss different of previous - current wrt to global test data

        fr_data_dist = []
        for model, loss, data in zip(local_models, loss_diffs, data_dist):
            model_q = compute_model_quality_with_loss_diff(loss, data)
            weight_vector = convert_model_dict_to_array(model)
            fr_data_dist.append([weight_vector, model_q])

        # fr_data_dist = torch.Tensor(fr_data_dist)
        
        if torch.cuda.is_available():
            self.dagmm.cuda()

        iter_ctr = 0
        for e in range(self.num_epochs):
            tmp_heat_values = []
            for i, (input_data) in enumerate(fr_data_dist):
                iter_ctr += 1

                self.dagmm.train()
                enc, dec, z, gamma = self.dagmm(input_data)

                total_loss, sample_energy, recon_error, cov_diag = self.dagmm.loss_function(input_data, dec, z, gamma, self.lambda_energy, self.lambda_cov_diag)

                self.dagmm.zero_grad()
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.dagmm.parameters(), 5)
                self.optimizer.step()
                tmp_heat_values.append(sample_energy.item())

            self.heat_values.append(tmp_heat_values) # append heat values per each epoch

        fr_idx = self.return_fr_index_numbers()
        return fr_idx

    def return_fr_index_numbers(self):
        last_round_heats = self.heat_values[-1]
        mean = stat.mean(last_round_heats)

        fr_idx = []

        for i, val in enumerate(last_round_heats):
            if val > mean:
                fr_idx.append(i)

        return fr_idx

class FreeRiders:
    def __init__(self):
        self.adv_delta_gaussian_mean = 0
        self.adv_delta_gaussian_std = 0.001
        self.linear_noising_std = 0.001
        self.std_multiplicator = 1
        self.power = 1

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def advance_delta_weights_attack(self, glob_model): #https://arxiv.org/pdf/1911.12560.pdf
        changing_model = deepcopy(glob_model).to(self.device)
        for idx, layer_tensor in enumerate(changing_model.parameters()):
            noise_additive = torch.randn_like(layer_tensor) * self.adv_delta_gaussian_std + self.adv_delta_gaussian_mean
            layer_tensor.data += noise_additive

        return changing_model

    def additive_noising_attack(self, glob_model, iteration): #https://proceedings.mlr.press/v130/fraboni21a.html
        changing_model = deepcopy(glob_model).to(self.device)
        if iteration != 0:
            for idx, layer_tensor in enumerate(changing_model.parameters()):
                mean_0 = torch.zeros(layer_tensor.size())
                std_tensor = torch.zeros(layer_tensor.size()) + self.std_multiplicator * self.linear_noising_std * iteration ** (-self.power)
                noise_additive = torch.normal(mean=mean_0, std=std_tensor).to(self.device)
                layer_tensor.data += noise_additive

        return changing_model