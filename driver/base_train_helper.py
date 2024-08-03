# -*- coding: utf-8 -*-
# @Time    : 19/11/8 9:54
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : base_syn_helper.py
from commons.utils import *
from mscv.summary import create_summary_writer
import torch
import matplotlib.pyplot as plt

from driver import std, mean
from stone_data.StoneDataLoader import get_full_image_list
from sklearn.model_selection import KFold
from driver import OPTIM
from models.EMA import EMA, MeanTeacher
from torch.cuda import empty_cache
from models import MODELS

plt.rcParams.update({'figure.max_open_warning': 20})


class BaseTrainHelper(object):
    def __init__(self, criterions, config):
        self.criterions = criterions
        self.config = config
        # p = next(filter(lambda p: p.requires_grad, generator.parameters()))
        self.use_cuda = config.use_cuda
        # self.device = p.get_device() if self.use_cuda else None
        self.device = config.gpu if self.use_cuda else None
        if self.config.train:
            self.make_dirs()
        self.define_log()
        self.reset_model()
        self.out_put_summary()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor
        self.init_params()

    def make_dirs(self):
        if not isdir(self.config.tmp_dir):
            os.makedirs(self.config.tmp_dir)
        if not isdir(self.config.save_dir):
            os.makedirs(self.config.save_dir)
        if not isdir(self.config.save_model_path):
            os.makedirs(self.config.save_model_path)
        if not isdir(self.config.tensorboard_dir):
            os.makedirs(self.config.tensorboard_dir)
        if not isdir(self.config.submission_dir):
            os.makedirs(self.config.submission_dir)
        code_path = join(self.config.submission_dir, 'code')
        if os.path.exists(code_path):
            shutil.rmtree(code_path)
        print(os.getcwd())
        shutil.copytree('../../', code_path, ignore=shutil.ignore_patterns('.git', '__pycache__', '*log*', '*tmp*'))

    def define_log(self):
        now = datetime.now()  # current date and time
        date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
        if self.config.train:
            log_s = self.config.log_file[:self.config.log_file.rfind('.txt')]
            self.log = Logger(log_s + '_' + str(date_time) + '.txt')
        else:
            self.log = Logger(join(self.config.save_dir, 'test_log_%s.txt' % (str(date_time))))
        sys.stdout = self.log

    def move_to_cuda(self):
        if self.use_cuda and self.model:
            torch.cuda.set_device(self.config.gpu)
            self.equipment = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.equipment)
            for key in self.criterions.keys():
                print(key)
                self.criterions[key].to(self.equipment)
            if len(self.config.gpu_count) > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=self.config.gpu_count)
        else:
            self.equipment = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save_checkpoint(self, state, filename='checkpoint.pth'):
        torch.save(state, filename)

    def count_parameters(self, net):
        model_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def save_model_checkpoint(self, epoch, optimizer):
        save_file = join(self.config.save_model_path, 'checkpoint_epoch_%03d.pth' % (epoch))
        self.save_checkpoint({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, filename=save_file)

    def save_last_checkpoint(self, model_optimizer=None, save_model=False, fold=0):
        opti_file_path = join(self.config.save_model_path, "fold_" + str(fold) + "_last_optim.opt")
        save_model_path = join(self.config.save_model_path, "fold_" + str(fold) + "_last_model.pt")
        ema_opti_file_path = join(self.config.save_model_path, "fold_" + str(fold) + "_last_ema.pt")
        if save_model:
            torch.save(self.model.state_dict(), save_model_path)
            if hasattr(self, 'ema'):
                torch.save(self.ema, ema_opti_file_path)

        if model_optimizer is not None:
            torch.save(model_optimizer.state_dict(), opti_file_path)

    def get_last_checkpoint(self, model_optimizer=None, load_model=False, fold=0):
        if model_optimizer is not None:
            load_file = join(self.config.save_model_path, "fold_" + str(fold) + "_last_optim.opt")
        if load_model:
            load_file = join(self.config.save_model_path, "fold_" + str(fold) + "_last_model.pt")
            # load_file = join(self.config.save_model_path, "iter_0_127_best_model.pt")
        print('loaded' + load_file)
        return torch.load(load_file, map_location=('cuda:' + str(self.device)))

    def save_best_checkpoint(self, model_optimizer=None, save_model=False, fold=0):
        opti_file_path = join(self.config.save_model_path, "fold_" + str(fold) + "_best_optim.opt")
        save_model_path = join(self.config.save_model_path, "fold_" + str(fold) + "_best_model.pt")
        ema_opti_file_path = join(self.config.save_model_path, "fold_" + str(fold) + "_best_ema.pt")
        if save_model:
            torch.save(self.model.state_dict(), save_model_path)
            if hasattr(self, 'ema'):
                torch.save(self.ema, ema_opti_file_path)

        if model_optimizer is not None:
            torch.save(model_optimizer.state_dict(), opti_file_path)

    def get_best_checkpoint(self, model_optimizer=None, load_model=False, fold=0):
        if model_optimizer is not None:
            load_file = join(self.config.save_model_path, "fold_" + str(fold) + "_best_optim.opt")
        if load_model:
            load_file = join(self.config.save_model_path, "fold_" + str(fold) + "_best_model.pt")
            # load_file = join(self.config.save_model_path, "fold_0_127_best_model.pt")
        print('loaded' + load_file)
        return torch.load(load_file, map_location=('cuda:' + str(self.device)))

    def load_best_optim(self, optim, fold=0):
        state_dict_file = self.get_best_checkpoint(model_optimizer=True, fold=fold)
        optim.load_state_dict(state_dict_file)
        return optim

    def load_best_state(self, fold=0):
        state_dict_file = self.get_best_checkpoint(load_model=True, fold=fold)
        net_dict = self.model.state_dict()
        net_dict.update(state_dict_file)
        self.model.load_state_dict(net_dict)
        del state_dict_file
        del net_dict
        if hasattr(self, 'ema'):
            del self.ema
            ema_opti_file_path = join(self.config.save_model_path, "fold_" + str(fold) + "_best_ema.pt")
            print('load ' + ema_opti_file_path)
            self.ema = torch.load(ema_opti_file_path, map_location=('cuda:' + str(self.device)))

    def load_last_state(self, fold=0):
        state_dict_file = self.get_last_checkpoint(load_model=True, fold=fold)
        net_dict = self.model.state_dict()
        net_dict.update(state_dict_file)
        self.model.load_state_dict(net_dict)
        del state_dict_file
        del net_dict
        if self.config.ema_decay > 0:
            if hasattr(self, 'ema'):
                del self.ema
            ema_opti_file_path = join(self.config.save_model_path, "fold_" + str(fold) + "_last_ema.pt")
            print('load ema' + ema_opti_file_path)
            self.ema = torch.load(ema_opti_file_path, map_location=('cuda:' + str(self.device)))

    def out_put_summary(self):
        self.summary_writer = create_summary_writer(self.config.tensorboard_dir)
        print('Model has param %.2fM' % (self.count_parameters(self.model) / 1000000.0))
        # if self.config.train:
        #     print(self.model)
        # summary(self.model.cpu(),
        #         torch.zeros((1, 3, self.config.patch_x, self.config.patch_y)))
        # x = torch.zeros((1, 3, self.config.patch_x, self.config.patch_y)).requires_grad_(False)
        # prediction = self.model(x)
        # g = make_dot(prediction)
        # save_path = join(self.config.save_dir, 'graph')
        # g.render(save_path, view=False)

    def write_summary(self, epoch, fold, criterions):
        for key in criterions.keys():
            self.summary_writer.add_scalar(
                key, criterions[key], epoch)

    def adjust_learning_rate_g(self, optimizer, i_iter, num_steps, istuning=False):
        warmup_iter = num_steps // 20
        if i_iter < warmup_iter:
            lr = self.lr_warmup(self.config.learning_rate, i_iter, warmup_iter)
        else:
            lr = self.lr_poly(self.config.learning_rate, i_iter, num_steps, 0.9)
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            if istuning:
                optimizer.param_groups[1]['lr'] = lr
            else:
                optimizer.param_groups[1]['lr'] = lr * 10


    def lr_poly(self, base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    def lr_warmup(self, base_lr, iter, warmup_iter):
        return base_lr * (float(iter) / warmup_iter)

    def save_vis_prob(self, images, save_dir, image_name, prob_maps, pred_labeled, label_img, ori_h, ori_w):
        grid = make_grid(images, nrow=1, padding=2)
        save_img = np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)) * std + mean
        save_img = np.clip(save_img * 255 + 0.5, 0, 255)
        visualize(save_img, join(save_dir, image_name + '_images'))
        visualize(np.expand_dims(prob_maps[1, :, :] / np.max(prob_maps[1, :, :]), -1),
                  '{:s}/{:s}_prob_inside'.format(save_dir, image_name))
        visualize(np.expand_dims(prob_maps[2, :, :] / np.max(prob_maps[2, :, :]), -1),
                  '{:s}/{:s}_prob_contour'.format(save_dir, image_name))

        # final_pred = Image.fromarray((pred_labeled * 255).astype(np.uint16))
        # final_pred.save('{:s}/{:s}_seg.png'.format(save_dir, image_name))

        # save colored objects
        pred_colored = np.zeros((ori_h, ori_w, 3))
        for k in range(1, pred_labeled.max() + 1):
            pred_colored[pred_labeled == k, :] = np.array(get_random_color())
        visualize(pred_colored, '{:s}/{:s}_seg_colored'.format(save_dir, image_name))
        pred_colored = np.zeros((ori_h, ori_w, 3))
        label_img = label_img == 1
        label_img = mlabel(label_img)
        for k in range(1, label_img.max() + 1):
            pred_colored[label_img == k, :] = np.array(get_random_color())
        visualize(pred_colored, '{:s}/{:s}_gt_colored'.format(save_dir, image_name))

    def get_n_fold(self, image_paths, fold, seed=666):
        kf = KFold(n_splits=self.config.nfold, random_state=seed, shuffle=True)
        cur_fold = 0
        for train_index, test_index in kf.split(image_paths):
            if cur_fold == fold:
                return train_index, test_index
            cur_fold += 1

    def get_image_full_list(self, seed=666):
        image_paths = get_full_image_list(dir=self.config.data_path)
        np.random.seed(seed)
        np.random.shuffle(image_paths)
        return image_paths

    def read_data(self, dataloader):
        while True:
            for batch in dataloader:
                image = batch['image_patch']
                label = batch['image_label']
                cat = batch['image_cat']
                name = batch['image_name']
                path = batch['image_path']
                text = batch['image_text']
                yield {
                    "image_patch": image,
                    "image_cat": cat,
                    "image_label": label,
                    "image_name": name,
                    "image_path": path,
                    "image_text": text,
                }

    def reset_optim(self):
        optimizer = OPTIM[self.config.learning_algorithm](
            self.model.optim_parameters(self.config.learning_rate),
            lr=self.config.learning_rate, weight_decay=5e-4)
        return optimizer

    def create_mtc(self, decay=0.999):
        if hasattr(self, 'ema'):
            del self.ema
        mtc = self.create_meteacher()
        for param in mtc.parameters():
            param.detach_()
        self.ema = MeanTeacher(mtc, decay)

    def create_model(self):
        mm = MODELS[self.config.model](backbone=self.config.backbone, n_channels=self.config.n_channels,
                                       num_classes=self.config.classes, pretrained=True)
        return mm

    def create_meteacher(self):
        mtc = self.create_model()
        mtc.to(self.device)
        return mtc

    def reset_model(self):
        if hasattr(self, 'model'):
            del self.model
            empty_cache()
        print("Creating models....")
        self.model = self.create_model()
        self.model.to(self.device)

    def correct_predictions(self, output_probabilities, targets):
        """
        Compute the number of predictions that match some target classes in the
        output of a model.

        Args:
            output_probabilities: A tensor of probabilities for different output
                classes.
            targets: The indices of the actual target classes.

        Returns:
            The number of correct predictions in 'output_probabilities'.
        """
        _, out_classes = output_probabilities.max(dim=1)
        correct = (out_classes == targets).sum()
        total_num = torch.prod(torch.tensor(targets.size())).float()
        return (correct.item() / total_num).float()

    def accuracy_check(self, mask, prediction):
        ims = [mask, prediction]
        np_ims = []
        for item in ims:
            if 'PIL' in str(type(item)):
                item = np.array(item)
            elif 'torch' in str(type(item)):
                item = item.cpu().detach().numpy()
            np_ims.append(item)
        compare = np.equal(np.where(np_ims[0] > 0.5, 1, 0), np_ims[1])
        accuracy = np.sum(compare)
        return accuracy / len(np_ims[0].flatten())
