import matplotlib
import sys

sys.path.extend(["../../", "../", "./"])
from commons.utils import *
import torch
from torch.cuda import empty_cache
import random
from driver.det_driver.BiClsHelper import BiClsHelper
from driver.Config import Configurable
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss
from torch.nn.utils import clip_grad_norm_
from module.torchcam.methods import GradCAM
from driver import transform_test
from copy import deepcopy
from torchvision.transforms.functional import to_pil_image
from module.torchcam.utils import overlay_mask
from commons.evaluation import calculate_metrics_multi
from stone_data import COLOR_DICT
from torch.utils.tensorboard import SummaryWriter
from module.losses import SoftDiceLoss

matplotlib.use("Agg")
import configparser

config = configparser.ConfigParser()
import argparse

log_template = "[Epoch %d/%d] [cls loss: %f] [cls acc: %f]"

from torchvision.models import resnet34
model_cam_extract = resnet34(pretrained=True)
cam_extractor = None


def main(config, seed=666):
    criterion = {
        'cls_loss': CrossEntropyLoss(),
        'seg_loss': SoftDiceLoss(),
        'con_loss': CosineEmbeddingLoss()
    }
    cls_help = BiClsHelper(criterion,
                         config)
    cls_help.move_to_cuda()
    print("data name ", cls_help.config.data_name)
    print("data patch x ", cls_help.config.patch_x)
    print("Random dataset Seed: %d" % (seed))

    start_fold = 0
    all_test_results = []
    final_results = {}
    for fold in range(start_fold, cls_help.config.nfold):
        print("**********Start train fold %d: **********" % (fold))
        if cls_help.config.fp16:
            from torch.cuda.amp import GradScaler as GradScaler
            scaler = GradScaler()
        else:
            scaler = None
        train_dataloader, vali_loader, test_image_paths = cls_help.get_pkldata_loader(fold=fold, seed=seed)
        best_acc = 0
        min_valid_loss = float('inf')
        bad_step = 0
        cls_help.log.flush()
        cls_help.reset_model()
        optimizer = cls_help.reset_optim()
        cls_help.create_mtc(decay=cls_help.config.ema_decay)
        writer = SummaryWriter(cls_help.config.tensorboard_dir)
        for epoch in range(cls_help.config.epochs):
            train(cls_help, train_dataloader, optimizer, epoch, scaler, writer)

            cls_help.log.flush()
            opt_s = ''
            for g in optimizer.param_groups:
                opt_s += "optimizer current_lr to %.8f \t" % (g['lr'])
            print(opt_s)
            vali_critics = valid(cls_help, vali_loader)
            if vali_critics['vali/acc'] > best_acc:
                print(" * Best vali acc at epoch %d: history = %.4f, current = %.4f" % (epoch, best_acc,
                                                                                        vali_critics['vali/acc']))
                test_pklpatch(cls_help, test_image_paths, cls_help.config.epochs, fold)
                best_acc = vali_critics['vali/acc']
                cls_help.write_summary(epoch, fold, vali_critics)
                cls_help.save_best_checkpoint(model_optimizer=optimizer, save_model=True, fold=fold)
            if vali_critics['vali/loss'] < min_valid_loss:
                min_valid_loss = vali_critics['vali/loss']
                bad_step = 0
            else:
                bad_step += 1
                if bad_step >= cls_help.config.patience:
                    test_pklpatch(cls_help, test_image_paths, cls_help.config.epochs, fold)
                    cls_help.save_last_checkpoint(model_optimizer=optimizer, save_model=True, fold=fold)
                    break
            if epoch == cls_help.config.epochs - 1:
                test_pklpatch(cls_help, test_image_paths, cls_help.config.epochs, fold)
            cls_help.save_last_checkpoint(model_optimizer=optimizer, save_model=True, fold=fold)
    np.save(cls_help.config.tmp_dir + "/" + config.model + "_result.npy", np.array(final_results))

    cls_help.log.flush()
    cls_help.summary_writer.close()

def train_seg(cls_help, train_dataloader, optimizer, epoch, total_epoch):
    cls_help.model.train()
    results = None
    optimizer.zero_grad()
    batch_num = int(np.ceil(len(train_dataloader.dataset) / float(cls_help.config.train_det_batch_size)))
    total_iter = batch_num * cls_help.config.epochs

    for i, batch in enumerate(train_dataloader):
        cls_help.adjust_learning_rate_g(optimizer, epoch * batch_num + i, total_iter, istuning=False)
        images, images_pair, segment, segment_pair, image_cat, image_cat_pair, image_text, image_text_pair, image_name, origin_img = cls_help.generate_batch(batch)

        seg_logits, seg_hx = cls_help.model(images, mode='seg')
        loss_seg = cls_help.criterions['seg_loss'](seg_logits, segment)

        optimizer.zero_grad()
        loss_seg.backward()
        optimizer.step()

        result = [loss_seg.item()]
        if results is None:
            results = Averagvalue(len([*result]))
        results.update(result, images.size(0))

    print('[Epoch {:d}/{:d}] Train Avg:'
          ' [seg_Loss {r[0]:.4f}]'.format(epoch, total_epoch, r=results.avg))
    empty_cache()
    return {
        'train/seg_loss': results.avg[0]
    }


def train(cls_help, train_dataloader, optimizer, epoch, scaler, writer):
    cls_help.model.train()
    results = None
    optimizer.zero_grad()
    batch_num = int(np.ceil(len(train_dataloader.dataset) / float(cls_help.config.train_det_batch_size)))
    total_iter = batch_num * cls_help.config.epochs

    for i, batch in enumerate(train_dataloader):
        cls_help.adjust_learning_rate_g(optimizer, epoch * batch_num + i, total_iter, istuning=False)
        images, images_pair, segment, segment_pair, image_cat, image_cat_pair, image_text, image_text_pair, image_name, myloss_target = cls_help.generat_batch(batch)

        seg_logits, seg_hx = cls_help.model(images, mode='seg')
        cls_logits, ipca1, ipca2 = cls_help.model(images, seg_hx, text=image_text, pair_x=images_pair, pair_text=image_text_pair, mode='bicls')
        loss_con = cls_help.criterions['con_loss'](ipca1, ipca2, image_cat_pair)
        loss_seg = cls_help.criterions['seg_loss'](seg_logits, segment)

        loss_cls = cls_help.criterions['cls_loss'](cls_logits, image_cat)

        probs = F.softmax(cls_logits)
        acc = cls_help.correct_predictions(probs, image_cat)

        loss = loss_cls + 0.1 * loss_seg + 0.1 * loss_con

        loss.backward()
        result = [loss_cls.item(), acc.item()]
        if results is None:
            results = Averagvalue(len([*result]))
        results.update(result, images.size(0))
        if (i + 1) % cls_help.config.update_every == 0 or i == batch_num - 1:
            clip_grad_norm_(filter(lambda p: p.requires_grad, cls_help.model.parameters()),
                            max_norm=cls_help.config.clip)
            optimizer.step()
            if hasattr(cls_help, 'ema'):
                cls_help.ema.update(cls_help.model)
            optimizer.zero_grad()
            empty_cache()
    print('[Epoch {:d}/{:d}] Train Avg:'
          ' [Loss Cls {r[0]:.4f}]'.format(epoch, cls_help.config.epochs, r=results.avg),
          ' [Acc {r[1]:.4f}]'.format(epoch, cls_help.config.epochs, r=results.avg))
    empty_cache()
    return {
        'train/cls_loss': results.avg[0]
    }

def valid(cls_help, vali_loader):
    cls_help.model.eval()
    cls_help.ema.model.eval()
    results = None
    cls_imgs_gt = []
    cls_imgs_pred = []
    for i, batch in enumerate(vali_loader):
        images, images_pair, segment, segment_pair, image_cat, image_cat_pair, image_text, image_text_pair, image_name, myloss_target = cls_help.generate_batch(batch)

        seg_logits, seg_hx = cls_help.model(images, mode='seg')
        cls_logits = cls_help.model(images, seg_hx, image_text)
        visual_batch(seg_logits, cls_help.config.submission_dir,
                         str(image_name) + '_' + str(image_cat.cpu().numpy()) + '_seg', channel=1, nrow=4)

        loss_cls = cls_help.criterions['cls_loss'](cls_logits, image_cat)

        probs = F.softmax(cls_logits)
        visual_batch(images, cls_help.config.submission_dir,
                     str(image_name) + '_' + str(image_cat.cpu().numpy()) + '_img', channel=3, nrow=4)
        visual_batch(segment, cls_help.config.submission_dir,
                     str(image_name) + '_' + str(image_cat.cpu().numpy()) + '_gt', channel=1, nrow=4)

        _, pred_cls = probs.max(dim=1)

        cls_imgs_gt.append(image_cat)
        cls_imgs_pred.append(pred_cls.item())
        acc = cls_help.correct_predictions(probs, image_cat)
        result = [acc.item(), loss_cls.item()]
        if results is None:
            results = Averagvalue(len([*result]))
        results.update(result, images.size(0))
    empty_cache()
    return {
        'vali/acc': results.avg[0],
        'vali/loss': results.avg[1],
        # 'vali/sensitivity': sensitivity,
    }


gradient = None
def hook_function(module, grad_input, grad_output):
    global gradient
    gradient = grad_output[0]
    # gradient = grad_input[0]

def test_pklpatch(cls_help, test_image_paths, epoch, fold):
    model_clone = deepcopy(cls_help.model)
    model_clone.eval()
    cls_help.ema.model.eval()
    results = None
    hook = model_clone.seg_net.dblock.register_backward_hook(hook_function)
    backbone_cam_extractor = GradCAM(model_clone, target_layer="backbone.layer4")
    GLCA_cam_extractor = GradCAM(model_clone, target_layer="fusion")
    fusion_cam_extractor = GradCAM(model_clone, target_layer="ITFM")
    save_dir = join(cls_help.config.submission_dir, 'test_fold_%d_epoch_%d' % (fold, epoch))
    if not exists(save_dir):
        makedirs(save_dir)
    shutil.rmtree(save_dir)
    makedirs(save_dir)
    all_patches_tag = []
    cls_imgs_gt = []
    cls_imgs_pred = []
    cls_imgs_score = []
    feats = []
    for dfs in test_image_paths:
        file_name = dfs['image_name']
        img = dfs['image_patch']
        rect_list = dfs['image_label']
        image_text = dfs['image_text']
        gt_cls = int(file_name[-7])

        segment = np.zeros((cls_help.config.patch_x, cls_help.config.patch_y))
        for i in range(len(rect_list)):
            segment[rect_list[i][0]:rect_list[i][2], rect_list[i][1]:rect_list[i][3]] = 1

        image_patch = transform_test(np.transpose(img, (1, 2, 0)).astype(np.uint8)).to(cls_help.device)
        image_patch = torch.unsqueeze(image_patch, dim=0)

        segment = torch.from_numpy(np.expand_dims(segment, 0)).to(cls_help.device).float()
        segment = torch.unsqueeze(segment, dim=0)

        image_text = torch.tensor(image_text[:-3]).to(cls_help.device).float()
        image_text = torch.unsqueeze(image_text, dim=0)


        seg_logits, seg_hx = model_clone(image_patch, mode='seg')
        logits = model_clone(image_patch, seg_hx, image_text)

        probs = F.softmax(logits)
        _, pred_cls = probs.max(dim=1)

        cls_imgs_gt.append(gt_cls)
        cls_imgs_pred.append(pred_cls.item())
        cls_imgs_score.append(probs[0].detach().cpu().numpy())


        visual_batch(image_patch, save_dir,
                     '%s_pred%d_gt%d_img' % (str(file_name), pred_cls.item(), gt_cls), channel=3,
                     nrow=4)

        # cam可视化
        backbone_activation_map = backbone_cam_extractor(logits.squeeze(0).argmax().item(), logits, retain_graph=True)
        backbone_result = overlay_mask(to_pil_image(np.transpose(img, (1, 2, 0)).astype(np.uint8)),
                                       to_pil_image(backbone_activation_map[0], mode='F'),
                                       alpha=0.7)
        backbone_result.save(join(save_dir, '%s_backbone_cam_heatmap.png' % (str(file_name))))


    eval = calculate_metrics_multi(np.array(cls_imgs_gt), np.array(cls_imgs_pred),
                                   dict=np.arange(len(COLOR_DICT) + 1))
    print(" * Test fold %d epoch %d" % (fold, epoch))
    print(" Total patches %d, positive examples %d. " % (
        len(all_patches_tag), np.sum(np.where(np.array(all_patches_tag) == 0, 0, 1))))
    print('\taccuracy\tprecision\trecall\tf1_score\tsensitivity\tspecificity\t\tAUC\t\tP_value')
    total = []
    for key in np.arange(len(COLOR_DICT) + 1):
        print('%d\t%0.4f\t\t%0.4f\t\t%0.4f\t\t%0.4f\t\t%0.4f\t\t%0.4f\t\t%0.4f\t\t%0.6f' % (key,
                                                                          eval[key][0], eval[key][1], eval[key][2],
                                                                          eval[key][3],
                                                                          eval[key][4], eval[key][5], eval[key][6], eval[key][7]))
        if key == 0:
            continue
        total.append(eval[key])
    total = np.array(total)
    print('avg:%0.4f\t\t%0.4f\t\t%0.4f\t\t%0.4f\t\t%0.4f\t\t%0.4f\t\t%0.4f\t\t%0.6f' % (
        np.mean(total[:, 0]), np.mean(total[:, 1]), np.mean(total[:, 2]), np.mean(total[:, 3]), np.mean(total[:, 4]),
        np.mean(total[:, 5]), np.mean(total[:, 6]), np.mean(total[:, 7])))
    weight_average = eval['weight_average']
    print('wavg%0.4f\t\t%0.4f\t\t%0.4f\t\t%0.4f\t\t%0.4f\t\t%0.4f\t\t%0.4f\t\t%0.6f' % (
        weight_average[0], weight_average[1], weight_average[2], weight_average[3], weight_average[4],
        weight_average[5], weight_average[6], weight_average[7]))

    zipDir(save_dir, save_dir + '.zip')
    shutil.rmtree(save_dir)
    empty_cache()
    print(" Total patches %d, positive examples %d. " % (
        len(all_patches_tag), np.sum(np.where(np.array(all_patches_tag) == 0, 0, 1))))



if __name__ == '__main__':
    torch.manual_seed(6666)
    torch.cuda.manual_seed(6666)
    random.seed(6666)
    np.random.seed(6666)
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    torch.backends.cudnn.benchmark = True  # cudn
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config-file', default='cls_configuration.txt')
    argparser.add_argument('--use-cuda', action='store_true', default=True)
    argparser.add_argument('--train', help='test not need write', default=True)
    argparser.add_argument('--gpu', help='GPU 0,1,2,3', default=0)
    argparser.add_argument('--gpu-count', help='number of GPUs (0,1,2,3)', default='0')
    argparser.add_argument('--run-num', help='run num: 0,2,3', default="0")
    argparser.add_argument('--ema-decay', help='ema decay', default="0")
    argparser.add_argument('--active-iter', help='active iter', default="1")
    argparser.add_argument('--seed', help='random seed', default=666, type=int)
    argparser.add_argument('--model', help='model name', default="resnet34")
    argparser.add_argument('--backbone', help='backbone name', default="resnet18")

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args, extra_args)
    torch.set_num_threads(config.workers + 1)
    config.train = args.train
    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)
    main(config, seed=args.seed)
