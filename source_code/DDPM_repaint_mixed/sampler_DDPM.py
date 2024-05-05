import torchvision
import math
from torch import nn
from torch.nn import functional as F
import os
import cv2
import torch
from torch.optim.lr_scheduler import _LRScheduler

print("true") if torch.cuda.is_available() else print("not")
device='cuda'

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = None
        self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(p=keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]

        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class ConditionalEmbedding(nn.Module):
    def __init__(self, num_labels, d_model, dim):
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.condEmbedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(in_ch, in_ch, 5, stride=2, padding=2)

    def forward(self, x, temb, cemb):
        x = self.c1(x) + self.c2(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.t = nn.ConvTranspose2d(in_ch, in_ch, 5, 2, 2, 1)

    def forward(self, x, temb, cemb):
        _, _, H, W = x.shape
        x = self.t(x)
        x = self.c(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=True):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.cond_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x, temb, labels):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h += self.cond_proj(labels)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, T,dim, num_labels, ch, ch_mult, num_res_blocks, dropout):
        super().__init__()
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.cond_embedding = ConditionalEmbedding(num_labels, ch, tdim)
        self.head = nn.Conv2d(dim, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(
                    ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=False))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, dim, 3, stride=1, padding=1)
        )

    def forward(self, x, t, labels):
        # Timestep embedding
        temb = self.time_embedding(t)
        cemb = self.cond_embedding(labels)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb, cemb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb, cemb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb, cemb)
        h = self.tail(h)

        assert len(hs) == 0
        return h


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, labels):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
              extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        loss = F.mse_loss(self.model(x_t, t, labels), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w=0.):
        super().__init__()

        self.model = model
        self.T = T
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def p_mean_variance(self, x_t, t, labels):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps = self.model(x_t, t, labels)
        nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))
        eps = (1. + self.w) * eps - self.w * nonEps
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T, labels):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t=x_t, t=t, labels=labels)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)

def save_mid_img(image_xt,time_step):
    sampledImgs=image_xt*0.5+0.5
    image=sampledImgs[:,:3,:,:]
    sketch=sampledImgs[:,3:,:,:].repeat(1, 3, 1, 1)

    sketch_max = torch.max(sketch)
    sketch_min = torch.min(sketch)

    print("sketch的最大值:", sketch_max)
    print("sketch的最小值:", sketch_min)

    show_data = torch.cat((image, sketch), dim=2)
    #torch.save(show_data, 'ConditionSampledImgs/' +label+'/'+ filename + 'sample.pth')
    show_data = torchvision.utils.make_grid(show_data, nrow=4, padding=5)
    plt_file = torchvision.transforms.ToPILImage(mode='RGB')(show_data)
    plt_file.save('ConditionSampledImgs/' + f"x_0_{time_step}" + '_sketch.jpg')

class ConditionGaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w=0.):
        super().__init__()

        self.model = model
        self.T = T
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def p_mean_variance(self, x_t, t, labels):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps = self.model(x_t, t, labels)
        nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))
        eps = (1. + self.w) * eps - self.w * nonEps
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x,x_T,labels, beta_1, beta_T, T):
        """
        Algorithm 2.
        """
        alphas = 1. - torch.linspace(beta_1, beta_T, T).double()
        alphas_bar = torch.cumprod(alphas, dim=0)

        #image_xt -> image_xt
        image_xt = x_T
        N = 10

        #By adjusting this value, you can control the degree of mixing. When the mix value is 0, it switches to the regular repaint algorithm
        mixed_value=0.6
        for time_step in reversed(range(T)):
            print(time_step)

            #normal repaint
            if time_step>(T*mixed_value):
                for n in range(N):
                    # forward process sketch
                    #create a tensor full of time t
                    sketch_t = torch.full([x.size(0), ], time_step, device=device)

                    #generate the noise
                    sketch_noise = torch.randn_like(x.to(torch.float32)).to(device)

                    #calculate the noise of x(guidance) at time t
                    sketch_xt = (
                            extract(torch.sqrt(alphas_bar).to(device), sketch_t.to(device), x.shape) * x.to(device) +
                            extract(torch.sqrt(1. - alphas_bar).to(device), sketch_t.to(device), x.shape) * sketch_noise.to(
                        device))

                    image_xt,_= torch.split(image_xt, [3, 1], dim=1)

                    image_xt = torch.cat([image_xt,sketch_xt], dim=1)

                    t = image_xt.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                    mean, var = self.p_mean_variance(x_t=image_xt, t=t,labels=labels)

                    # no noise when t == 0
                    if time_step > 0:
                        noise = torch.randn_like(image_xt)
                    else:
                        noise = 0

                    image_xt = mean + torch.sqrt(var) * (noise)

                    if time_step > 0:
                        image_xt = (
                                extract(torch.sqrt(alphas).to(device), t.to(device), image_xt.shape) * image_xt.to(device) +
                                extract(torch.sqrt(1. - alphas).to(device), t.to(device),
                                        image_xt.shape) * torch.randn_like(image_xt).to(device))
            else:

                t = image_xt.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                mean, var = self.p_mean_variance(x_t=image_xt, t=t, labels=labels)

                # no noise when t == 0
                if time_step > 0:
                    noise = torch.randn_like(image_xt)
                else:
                    noise = 0

                image_xt = mean + torch.sqrt(var) * (noise)
                assert torch.isnan(image_xt).int().sum() == 0, "nan in tensor."
            #save processing
            # if time_step % 10 == 0:
            #     save_mid_img(image_xt,time_step)

        x_0 = image_xt
        return torch.clip(x_0, -1, 1)

class Dateconcat():
    def __init__(self, image_dataset, mask_dateset):
        self.image_dataset = image_dataset
        self.mask_dateset = mask_dateset

    def __len__(self):
        return len(self.mask_dateset)

    def __getitem__(self, index):
        image, label = self.image_dataset[index]
        mask, _ = self.mask_dateset[index]

        mask = mask[:1, :, :]
        image=(image-0.5)*2
        mask=(mask-0.5)*2
        return torch.cat((image, mask), dim=0), label


def NCsampler(modelConfig):
    device=modelConfig["device"]
    # model setup
    net_model = UNet(T=modelConfig["T"], dim=modelConfig["dim"],num_labels=modelConfig["num_label"], ch=modelConfig["channel"],
                     ch_mult=modelConfig["channel_mult"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["training_load_weight"]), map_location=device), strict=False)
        print("Model weight load down.")

    Conditionsampler=ConditionGaussianDiffusionSampler(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"],w = modelConfig["w"]).to(device)

    weight = torch.load("CheckpointsCondition/Sketch2img_ckpt_400_.pt", map_location=device)
    net_model.load_state_dict(weight['model_state_dict'])

    seen_label=['car','clothes','dog']
    num=[1,2,3]

    for i in [3]:
        label=i
        label_name=seen_label[i-1]
        folder_path='test_data/sketch/'+label_name+"/"
        files = os.listdir(folder_path)
        for filename in files:
            print(filename)
            with torch.no_grad():
                num_i=label
                labels = torch.full((modelConfig["batch_size"],), num_i).to(device)
                noisyImage = torch.randn(
                    size=[modelConfig["batch_size"], modelConfig["dim"], modelConfig["img_size"], modelConfig["img_size"]],
                    device=device)
                # ---just sampler---
                #sampledImgs = sampler(noisyImage,labels)

                # ---repaint sampler---
                folder = folder_path
                image_path = folder + filename
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (modelConfig["img_size"], modelConfig["img_size"]))
                guidance_batch = torch.tensor(image)
                guidance_batch=torch.unsqueeze(guidance_batch, 0)

                guidance_selected = torch.unsqueeze(guidance_batch, 0)
                guidance = guidance_selected.repeat((modelConfig["batch_size"], 1, 1, 1))
                guidance = (guidance.float()-127.5) / 127.5
                sampledImgs = Conditionsampler(guidance,noisyImage, labels,modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"])

                image=sampledImgs[:,:3,:,:]* 0.5 - 0.5
                sketch=sampledImgs[:,3:,:,:].repeat(1, 3, 1, 1)
                show_data = torch.cat((image, sketch), dim=2)
                torch.save(show_data, 'ConditionSampledImgs/'+label_name+'/'+filename+ 'data.pth')
                show_data = torchvision.utils.make_grid(show_data, nrow=4, padding=5)
                plt_file = torchvision.transforms.ToPILImage(mode='RGB')(show_data)
                plt_file.save('ConditionSampledImgs/'+label_name+'/' +filename+'.jpg')


def main(model_config=None):

    #parameter setting
    modelConfig = {
        "state": "train",  # or eval
        "epoch": 100,
        "batch_size": 4,
        "T": 500,
        "w": 3,
        "dim": 4,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 3,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 64,
        "num_label":3,
        "grad_clip": 1.,
        "device": "cuda",
        "save_dir": "./CheckpointsCondition/",
        "training_load_weight": None,
        "test_load_weight": "ckpt_63_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow": 8

    }
    if model_config is not None:
        modelConfig = model_config
    #train(modelConfig)
    NCsampler(modelConfig)


if __name__ == '__main__':
    main()
