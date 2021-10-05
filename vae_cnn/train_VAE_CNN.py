from argparse import ArgumentParser
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('..')
from base_module import BaseDynamicsModule
from models.VAE_CNN import VAE_CNN
from utilities.callbacks import BestValidationCallback, TestEndCallback
from utilities.toolsies import seed_everything, str2bool, none_or_float
from utilities.losses import kld_loss, geco_constraint
from utilities.losses import *
import skimage.metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from lpips_pytorch import LPIPS

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
def SSIM(x,y,L=1,k1=0.01,k2=0.03):
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    
    results = np.zeros((x.shape[0],))
    for i in range(x.shape[0]):
        if (x.shape[1]) > 1:
            results[i] = skimage.metrics.structural_similarity(x[i,:],y[i,:],multichannel=True)
        else:
            results[i] = skimage.metrics.structural_similarity(x[i,:].squeeze(),y[i,:].squeeze())
    return np.mean(results)
def PSNR(x,y):
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    
    results = np.zeros((x.shape[0],))
    for i in range(x.shape[0]):
        results[i] = skimage.metrics.peak_signal_noise_ratio(x[i,:],y[i,:])
    return np.mean(results)
def MSE(x,y):
    x = x.cpu()
    y = y.cpu()
    return torch.mean((x-y).pow(2))
class DynamicsVAE_CNN(BaseDynamicsModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = VAE_CNN(input_dim = self.hparams.model_input_size, 
                                output_dim = self.hparams.model_output_size,
                                fc_dim = self.hparams.model_fc_size,
                                latent_dim = self.hparams.model_latent_size,
                                hidden_dims = self.hparams.model_hidden_size, 
                                filter_size = 3,
                                padding = 1,
                                stride = 2,
                                nonlinearity = self.hparams.model_nonlinearity, 
                                dropout_pct = self.hparams.model_dropout_pct,
                                use_layer_norm = self.hparams.use_layer_norm)

        if self.hparams.kld_scaling_type=='geco':
            self.geco_multiplier = 1
            self.C_ma = None
        elif self.hparams.kld_scaling_type=='beta_fixed':
            self.beta = self.hparams.beta
        elif self.hparams.kld_scaling_type=='beta_anneal':
            self.beta = self.hparams.beta_initial
            self.beta_initial = self.hparams.beta_initial
            self.beta_max = self.hparams.beta_max
            self.beta_anneal_steps = self.hparams.beta_anneal_steps
        else:
            raise Exception(f'Wrong KLD scaling method: {self.hparams.kld_scaling_type}')

        if (self.hparams.use_supervision) and (self.hparams.sup_loss_type == 'sigmoid_parametrized'):
            self.w1 = torch.nn.Parameter(torch.tensor(1.0))
            self.w2 = torch.nn.Parameter(torch.tensor(1.0))

        self.val_rec_loss_sizes = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 
                                    170, 200, 250, 300,350 , 400, 450, 500]
        # self.test_rec_loss_sizes = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 
                                    # 170, 200, 250, 300,350, 400,450, 500]
        step=10
        self.test_rec_loss_sizes = list(range(step, 800+step, step))

        if self.hparams.model_output_size not in self.val_rec_loss_sizes:
            self.val_rec_loss_sizes.append(self.hparams.model_output_size)
        self.lpips_loss = None

    def rollout(self, batch, start, rollout_size):
        trajectory = batch['trajectory'].to(self.device)
        input_end_point = output_start_point = start + self.hparams.model_input_size
        input_trajectory = trajectory[:, start:input_end_point, :]
        output, _, _ = self.model(input_trajectory)
        model_input_size = self.hparams.model_input_size
        model_output_size = self.hparams.model_output_size

        while output.size(1) < rollout_size: #keep rolling till we reach the required size
            #if the model output is smaller than the input use previous data
            if model_output_size < model_input_size:
                keep_from_input = model_input_size - model_output_size
                input_trajectory = torch.cat((input_trajectory[:, -keep_from_input:, :], 
                                                output[:, -model_output_size:,:]), dim=1)
            else:
                input_trajectory = output[:, -model_input_size:, :]
            new_output = self.model(input_trajectory)[0]
            output = torch.cat((output, new_output), dim=1)
        if self.hparams.noise_std is not None:
            target = batch['target'].to(self.device)
            return output[:, :rollout_size, :], target[:, output_start_point:(output_start_point + 
                                                                            rollout_size), :]
        else:
            return output[:, :rollout_size, :], trajectory[:, output_start_point:(output_start_point + 
                                                                            rollout_size), :]


    def forward(self, batch):
        # one forward pass with the models default input output sizes
        # the starting point is randomized in here
        trajectory = batch['trajectory']
        start = self.get_start(batch, self.hparams.model_output_size)

        input_end_point = output_start_point = start + self.hparams.model_input_size
        input_trajectory = trajectory[:, start:input_end_point, :]
        
        target_trajectory = [None]*self.hparams.train_iteration
        output_trajectory = [None]*self.hparams.train_iteration
        mu = [None]*self.hparams.train_iteration
        logvar = [None]*self.hparams.train_iteration
        
        
        for i in range(self.hparams.train_iteration):
            target_trajectory[i] = trajectory[:, output_start_point+self.hparams.model_output_size*i:(output_start_point + 
                                                    self.hparams.model_output_size*(i+1)), :]
            
            
            output_trajectory_t, mu_t, logvar_t = self.model(input_trajectory)
            
            
            input_trajectory = output_trajectory_t

            output_trajectory[i] = output_trajectory_t
            mu[i] = mu_t
            logvar[i] = logvar_t
            
            
        
        return torch.cat(output_trajectory,dim=1), torch.cat(target_trajectory,dim=1), mu, logvar

    def get_normalized_labels(self, labels):
        # make lengths 0-1
        labels_norm = {}
        for k,v in labels.items():
            labels_norm[k] = (v-self.labels_min[k])/(self.labels_max[k] - self.labels_min[k] + 1e-6)
        return labels_norm

    def get_geco_C(self, rec_loss):
        tol = self.hparams.geco_tolerance
        alpha = self.hparams.geco_alpha

        C = geco_constraint(rec_loss, tol)
        C_curr = C.detach() # keep track for logging
        if self.C_ma is None:
            self.C_ma = C.detach()
        else:
            self.C_ma = alpha * self.C_ma + (1 - alpha) * C.detach()
        C = C + (self.C_ma - C.detach()) 

        return C, C_curr

    def update_geco_multiplier(self, C):
        # clamping the langrange multiplier to avoid inf values
        speed = self.hparams.geco_speed
        clipping = self.hparams.geco_clipping
        self.geco_multiplier = self.geco_multiplier * torch.exp(speed * C.detach())
        self.geco_multiplier = torch.clamp(self.geco_multiplier, 1.0/clipping, clipping)

    def get_label_loss(self, labels, mu):
        # loss for labels
        labels_min, labels_max = self.labels_min.to(self.device), self.labels_max.to(self.device)
        num_factors = len(labels_min)

        if self.hparams.sup_loss_type == 'sigmoid':
            pred_scaled = torch.sigmoid(mu[:, :num_factors]) \
                            * (labels_max - labels_min) + labels_min
            label_loss = F.l1_loss(pred_scaled, labels)
        elif self.hparams.sup_loss_type == 'sigmoid_parametrized':
            pred_scaled = self.w1 * torch.sigmoid(self.w2 * mu[:, :num_factors]) * \
                        (labels_max - labels_min) + labels_min
            self.log('train/sup/w2', self.w2, prog_bar=True)
            self.log('train/sup/w1', self.w1, prog_bar=True)
            label_loss = F.l1_loss(pred_scaled, labels)
        elif self.hparams.sup_loss_type == 'linear':
            pred_scaled = mu[:, :num_factors]
            label_loss = F.l1_loss(pred_scaled, labels)
        elif self.hparams.sup_loss_type == 'linear_scaled':
            pred_scaled = mu[:, :num_factors] * \
                            (labels_max - labels_min) + labels_min
            label_loss = F.l1_loss(pred_scaled, labels)
        elif self.hparams.sup_loss_type == 'BCE':
            labels_norm = (labels-labels_min)/(labels_max - labels_min + 1e-6)
            BCE = torch.nn.BCEWithLogitsLoss(reduction='mean')
            label_loss = BCE(mu[:, :num_factors], labels_norm)
        else:
            raise Warning('Wrong supervised loss type: ', self.hparams.sup_loss_type)
        return label_loss

    def update_label_loss_multiplier(self):
        pass

    def training_step(self, train_batch, batch_idx):
        if self.datasets['val'].noise is not None:
            self.datasets['val'].reset_noise()
            eprint('reset val dataset noise')
        rec_loss = 0.0
        kld = 0.0
        label_loss = 0.0
        for i in range(self.hparams.samples_per_batch_train):
            output_trajectory, target_trajectory, mu, logvar = self.forward(train_batch)
            rec_loss = rec_loss + self.reconstruction_loss(output_trajectory, target_trajectory)
            loss_normalizer = 1
            kld = kld + kld_loss(torch.cat(mu,dim=1), torch.cat(logvar,dim=1)) / loss_normalizer

            if self.hparams.use_supervision:
                labels = train_batch['labels']
                for j in range(self.hparams.train_iteration):
                    label_loss += label_loss + self.get_label_loss(labels, mu[j])

        rec_loss = rec_loss / self.hparams.samples_per_batch_train/self.hparams.train_iteration
        kld = kld / self.hparams.samples_per_batch_train/self.hparams.train_iteration
        self.log('train/rec', rec_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train/kld', kld, prog_bar=True, on_step=False, on_epoch=True)
        if self.hparams.use_supervision:
            label_loss = label_loss/self.hparams.samples_per_batch_train/self.hparams.train_iteration
            self.log('train/label_loss', label_loss, prog_bar=True, on_step=False, on_epoch=True)
            rec_loss = rec_loss + self.hparams.sup_multiplier * label_loss

        if self.hparams.kld_scaling_type=='geco':
            C, C_curr = self.get_geco_C(rec_loss)
            train_loss = self.geco_multiplier * C + kld
            self.log('train/geco/C', C_curr, prog_bar=False, on_step=False, on_epoch=True)
            self.log('train/geco/C_ma', self.C_ma.detach(), prog_bar=False, on_step=False, on_epoch=True)
            self.log('train/geco/geco_multiplier', self.geco_multiplier, prog_bar=True, on_step=False, on_epoch=True)
            self.update_geco_multiplier(C)
        elif self.hparams.kld_scaling_type=='beta_fixed':
            train_loss = rec_loss + self.beta * kld
            self.log('train/beta', self.beta, on_step=False, on_epoch=True)
        elif self.hparams.kld_scaling_type=='beta_anneal':
            train_loss = rec_loss + self.beta * kld
            new_beta = self.beta_initial + (self.beta_max - self.beta_initial)\
                                        *(self.global_step+1)/(self.beta_anneal_steps+1)
            self.beta = np.minimum(self.beta_max, new_beta)
            self.log('train/beta', self.beta, on_step=False, on_epoch=True)

        self.log('train/loss', train_loss, prog_bar=True, on_step=False, on_epoch=True)
        
        if (batch_idx % self.hparams.log_freq) == 0:
            self.log_rec_losses(train_batch, 'train', self.val_rec_loss_sizes,
                                                on_step=False, on_epoch=True)
            if self.hparams.debug:
                self.log_histogram("debug/mu", mu)
                self.log_histogram("debug/logvar", logvar)

        # print(f"{train_loss.item():.3f} {rec_loss.item():.3f} {kld.item():.3f} {self.beta:.3f}")
        return train_loss

    def validation_step(self, val_batch, batch_idx):
        if self.datasets['train'].noise is not None:
            self.datasets['train'].reset_noise()
            eprint('reset training dataset noise')
        
        random_start = self.hparams.use_random_start
        self.hparams.use_random_start = False
        self.log_rec_losses(val_batch, 'val', self.val_rec_loss_sizes)
        self.hparams.use_random_start = random_start
        label_loss = 0.0
        if self.hparams.use_supervision:
            _, _, mu, logvar = self.forward(val_batch)
            labels = val_batch['labels']
            for j in range(self.hparams.train_iteration):
                label_loss += label_loss + self.get_label_loss(labels, mu[j])
            self.log('val/label_loss', label_loss)

    def test_step(self, test_batch, batch_idx, dataloader_idx=None):
        if self.lpips_loss is None:
            self.lpips_loss = LPIPS(net_type=self.hparams.lpips_backend, version='0.1').to(self.device)

        for i in range(self.hparams.samples_per_batch_test):
            self.log_rec_losses(test_batch, 'test', self.test_rec_loss_sizes)

    def log_rec_losses(self, batch, stage, rec_loss_sizes, on_epoch=True, on_step=False):
        with torch.no_grad():
            # reconstruction losses for longer trajectories.
            max_rollout = np.max(rec_loss_sizes) 
            start = self.get_start(batch, rec_loss_sizes)
            # normalize per number of pixels
            preds, gts = self.rollout(batch, start=start, rollout_size=max_rollout)
            batch_size, time_steps, h, w = preds.size()
            
            if stage == 'test':
                metrics_c = {
                    'ssim':  [],
                    'psnr':  [],
                    'lpips': [],
                }
                
                for t in range(time_steps):
                    metrics_p = {k: [] for k in metrics_c.keys()}
                    
                    for b in range(batch_size):
                        pr = preds[b, t]
                        gt = gts[b, t]
                        metrics_p['lpips'].append(self.lpips_loss(gt, pr).item())        
                        pr=pr.cpu().numpy()
                        gt=gt.cpu().numpy()
                        metrics_p['ssim'].append(ssim(gt, pr))
                        metrics_p['psnr'].append(psnr(gt, pr))
                    
                    for k, mp in metrics_p.items():
                        metrics_c[k].append(mp)
                    
                    if t+1 in rec_loss_sizes:
                        # Cummulative metrics
                        for k, mc in metrics_c.items():
                            self.log(f'test/cumm/{k}/mean/{t+1:04d}',  np.mean(mc))
                            self.log(f'test/cumm/{k}/var/{t+1:04d}',   np.var(mc))
                        # Point metrics
                        for k, mp in metrics_p.items():
                            self.log(f'test/point/{k}/mean/{t+1:04d}',  np.mean(mp))
                            self.log(f'test/point/{k}/var/{t+1:04d}', np.var(mp))
    
            # Use unitary decoder std to allow comparison
            for ln in rec_loss_sizes:
                pred=preds[:,:ln].to(self.device)
                gt=gts[:,:ln].to(self.device)
                pixel_loss =  mse_loss(pred, gt)
                self.log(f'{stage}/cumm/{ln:04d}', pixel_loss)
                
                pixel_loss =  mse_loss(pred, gt, reduction='none').mean([0, 1]).sum()
                self.log(f'test/pixel_loss/{ln:04d}', pixel_loss)
                self.log(f'test/pixel_loss_comp/{ln:04d}', pixel_loss, 
                                         )
                self.log(f'test/cumm/mse/{ln:04d}', mse_loss(pred, gt) )
                self.log(f'test/point/mse/{ln:04d}', mse_loss(pred[-1], gt[-1]) )
            
            
    def get_start(self, batch, rec_loss_sizes):
        if self.hparams.use_random_start==True:
            length = batch['trajectory'].size(1)
            max_rollout = np.max(rec_loss_sizes) 
            max_start = length - self.hparams.model_input_size*self.hparams.train_iteration - max_rollout
            start = np.random.choice(range(max_start))
        else:
            start = 0
        return start

if __name__ == '__main__':

    # parametrize the network
    parser = ArgumentParser()
    parser.add_argument('--project_name', default='dummy')
    parser.add_argument('--model', default='vae_cnn')

    parser.add_argument('--dataset', type=str,default='pixel_pendulum-1')
    parser.add_argument('--dataset_dt', type=float, default=0.01)
    parser.add_argument('--noise_std', type=none_or_float, nargs='?', default=None)
    # L1, MSE
    parser.add_argument('--rec_loss_type', type=str, default='CNN_MSE')
    parser.add_argument('--batch_size', type=int, default = 32)
    parser.add_argument('--batch_size_val', type=int, default=16)
    parser.add_argument('--samples_per_batch_train', type=int, default=1)
    parser.add_argument('--samples_per_batch_val', type=int, default=10)
    parser.add_argument('--samples_per_batch_test', type=int, default=10)
    parser.add_argument('--use_random_start', type=str2bool,default=True)

    parser.add_argument('--model_nonlinearity', type=str, default='leaky')
    parser.add_argument('--model_input_size', type=int,default=5)
    parser.add_argument('--model_hidden_size', nargs='+', type=int,default=[16,16])
    parser.add_argument('--model_fc_size', type=int, nargs='+',default = [16])
    parser.add_argument('--model_latent_size', type=int, default = 5)
    parser.add_argument('--model_output_size', type=int, default = 5)
    parser.add_argument('--model_dropout_pct', type=float, default=0.0)
    parser.add_argument('--use_layer_norm', type=str2bool, default=True)
    # BETA
    parser.add_argument('--kld_scaling_type', type=str, default='beta_fixed')
    # beta_fixed
    parser.add_argument('--beta', type=float, default=1)
    # beta_anneal
    parser.add_argument('--beta_initial', type=float, default=None) #
    parser.add_argument('--beta_max', type=float, default=None)
    parser.add_argument('--beta_anneal_steps', type=float, default=None)
    # geco
    parser.add_argument('--geco_tolerance', type=float, default=None)
    parser.add_argument('--geco_alpha', type=float, default=None)
    parser.add_argument('--geco_speed', type=float, default=None)
    parser.add_argument('--geco_clipping', type=float, default=None)

    # SUPERVISION
    # sigmoid, sigmoid_parametrized, linear, linear_scaledled, BCE
    parser.add_argument('--use_supervision', type=str2bool, default=True)
    parser.add_argument('--sup_loss_type', type=str, default='sigmoid_parametrized') 
    parser.add_argument('--sup_multiplier', type=float, default=1e-2)

    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--scheduler_patience', type=int, default=50)
    parser.add_argument('--scheduler_factor', type=float, default=0.3)
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-7)
    parser.add_argument('--scheduler_threshold', type=float, default=1e-5)

    parser.add_argument('--gradient_clip_val', type=float,default=1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_epochs', type=int,default = 1000)
    parser.add_argument('--monitor', type=str, default='val/rec/cumm/0250')
    parser.add_argument('--early_stopping_patience', type=int, default = 300)
    parser.add_argument('--lpips_backend', type=str, default='alex', help='backend of the lpips loss. alex or vgg')

    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--time', type=int, default=0)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--use_wandb', type=str2bool, default=True)
    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--fast_dev_run', type=str2bool, default=False)
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--progress_bar_refresh_rate', type=int, default=1)
    
    parser.add_argument('--train_iteration', type=int, default=3)

    hparams = parser.parse_args()
    print(hparams)

    seed_everything(hparams.seed)
    pl.seed_everything(hparams.seed)
    model = DynamicsVAE_CNN(**vars(hparams))


    if hparams.use_wandb:
        save_dir = os.path.join(os.environ['WANDB_DIR'], hparams.project_name)
        os.makedirs(save_dir, exist_ok=True)
        logger = pl.loggers.WandbLogger(project=hparams.project_name, save_dir=save_dir)
        logger.log_hyperparams(vars(hparams))
        if hparams.debug:
            logger.watch(model, log='all', log_freq=hparams.log_freq)
        checkpoint_dir = os.path.join(logger.experiment.dir, 'checkpoints/')
    else:
        # log_dir = os.path.join(os.environ['EXP_DIR'], 'tensorboard')
        log_dir = './tensorboard/'
        print(f'Using tensorboard from {log_dir}')
        os.makedirs(os.path.join(log_dir, hparams.project_name), exist_ok=True)
        experiment_name = f'in_{hparams.model_input_size}_out_{hparams.model_output_size}'
        logger = pl.loggers.TensorBoardLogger(save_dir=log_dir, name=experiment_name)
        checkpoint_dir = logger.log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    print(f'Checkpoint dir {checkpoint_dir}')
 
    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    early_stop_callback = pl.callbacks.EarlyStopping(monitor=hparams.monitor, min_delta=0.00, 
                patience=hparams.early_stopping_patience, verbose=True, mode='min')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename='{epoch}',
                monitor=hparams.monitor, 
                save_top_k=1,verbose=True, mode='min',
                save_last=False)
    best_validation_callback = BestValidationCallback(hparams.monitor, hparams.use_wandb)
    test_end_callback = TestEndCallback(hparams.use_wandb)

    trainer = pl.Trainer.from_argparse_args(hparams, 
                logger=logger,
                log_every_n_steps=10,
                # track_grad_norm=2,
                gradient_clip_val=hparams.gradient_clip_val,
                
                callbacks=[ checkpoint_callback,
                            early_stop_callback, 
                            lr_monitor_callback, 
                            best_validation_callback,
                            test_end_callback
                        ],
                deterministic=True,
                progress_bar_refresh_rate=hparams.progress_bar_refresh_rate,
                check_val_every_n_epoch=1
                )

    trainer.fit(model)
    trainer.test()