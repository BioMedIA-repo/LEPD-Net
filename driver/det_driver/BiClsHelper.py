from commons.utils import *
import torch
from driver.base_train_helper import BaseTrainHelper
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from stone_data.StoneDataLoader import StoneBiClsDataSet, get_full_image_list, StoneBiClsPlkDataSet
from sklearn.model_selection import KFold
from driver import transform_local, transform_test
from driver import OPTIM
from models.EMA import EMA, MeanTeacher
from torch.cuda import empty_cache
from models import MODELS
from torch import nn
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize, resize, to_pil_image
from module.torchcam.utils import overlay_mask

plt.rcParams.update({'figure.max_open_warning': 20})

class BiClsHelper(BaseTrainHelper):
    def __init__(self, criterions, config):
        super(BiClsHelper, self).__init__(criterions, config)
        self.get_Kclass_queue = None

    def init_params(self):
        return

    def create_model(self):
        mm = MODELS[self.config.model](backbone=self.config.backbone, n_channels=self.config.n_channels,
                                       num_classes=self.config.classes, pretrained=True)
        return mm

    def generate_batch(self, batch):
        images = batch['image_patch'].to(self.equipment).float()
        segment = batch['image_label'].to(self.equipment).float()
        images_pair = batch['image_patch_pair'].to(self.equipment).float()
        segment_pair = batch['image_label_pair'].to(self.equipment).float()
        image_cat = batch['image_cat'].to(self.equipment).long()
        image_cat_pair = batch['image_cat_pair'].to(self.equipment).long()
        image_text = batch['image_text'].to(self.equipment).float()
        image_text_pair = batch['image_text_pair'].to(self.equipment).float()
        image_name = batch['image_name']

        myloss_target = batch['myloss_target'].to(self.equipment).float()
        return images, images_pair, segment, segment_pair, image_cat, image_cat_pair, image_text, image_text_pair, image_name, myloss_target

    def merge_batch(self, batch):
        image_patch = [torch.unsqueeze(inst["image_patch"], dim=0) for inst in batch]
        image_patch_pair = [torch.unsqueeze(inst["image_patch_pair"], dim=0) for inst in batch]
        image_segment = [torch.unsqueeze(inst["image_label"], dim=0) for inst in batch]
        image_segment = torch.cat(image_segment, dim=0)
        image_segment_pair = [torch.unsqueeze(inst["image_label_pair"], dim=0) for inst in batch]
        image_segment_pair = torch.cat(image_segment_pair, dim=0)
        image_cat = [inst["image_cat"] for inst in batch]
        image_cat_pair = [inst["image_cat_pair"] for inst in batch]
        image_text = [inst["image_text"] for inst in batch]
        image_text_pair = [inst["image_text_pair"] for inst in batch]
        image_patch = torch.cat(image_patch, dim=0)
        image_patch_pair = torch.cat(image_patch_pair, dim=0)
        image_cat = torch.tensor(image_cat)
        image_cat_pair = torch.tensor(image_cat_pair)
        image_text = torch.tensor(image_text)
        image_text_pair = torch.tensor(image_text_pair)
        image_name = [inst["image_name"] for inst in batch]

        myloss_target = [torch.unsqueeze(inst["myloss_target"], dim=0) for inst in batch]
        myloss_target = torch.cat(myloss_target, dim=0)
        return {"image_patch": image_patch,
                "image_patch_pair": image_patch_pair,
                "image_label": image_segment,
                "image_label_pair": image_segment_pair,
                "image_name": image_name,
                "image_cat": image_cat,
                "image_cat_pair": image_cat_pair,
                "image_text": image_text,
                "image_text_pair": image_text_pair,
                "myloss_target": myloss_target
                }

    def get_new_loader(self, fold, seed=777):
        image_paths = self.get_image_full_list(seed=seed)
        train_index, test_index = self.get_n_fold(image_paths=image_paths, fold=fold, seed=seed)


        train_image_paths = [image_paths[i] for i in train_index]
        test_image_paths = [image_paths[i] for i in test_index]
        train_img_names = [basename(p[0]) for p in train_image_paths]
        test_img_names = [basename(p[0]) for p in test_image_paths]

        print("Train images %d: " % (len(train_img_names)), train_img_names)
        print("Test images %d: " % (len(test_img_names)), test_img_names)

        data_path = '../../../image_patch/CLAHE_patch_percent6_text_seglabel'
        image_dfs = torch.load(join(data_path, 'stone_data.pkl'))
        new_train_image_paths = []
        new_test_image_paths = []
        for img in image_dfs:
            if img['image_name'].split('_')[0] + '.png' in train_img_names:
                new_train_image_paths.append(img)
            else:
                new_test_image_paths.append(img)
        train_dataset = StoneBiClsPlkDataSet(new_train_image_paths, input_size=(self.config.patch_x, self.config.patch_y),
                                             augment=transform_local,num_classes=self.config.classes,
                                             stone_label_smoothing=0.1)
        self.get_Kclass_queue = train_dataset.get_Kclass_queue
        self.getPairInfo = train_dataset.getPairInfo
        valid_dataset = StoneBiClsPlkDataSet(new_test_image_paths, input_size=(self.config.patch_x, self.config.patch_y),
                                             augment=transform_test, num_classes=self.config.classes,
                                             stone_label_smoothing=0.1)
        train_loader = DataLoader(train_dataset, batch_size=self.config.train_det_batch_size, shuffle=True,
                                  num_workers=self.config.workers,
                                  collate_fn=self.merge_batch,
                                  drop_last=True if len(train_image_paths) % self.config.train_det_batch_size != 0 else False)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                                  num_workers=0,
                                  collate_fn=self.merge_batch)
        return train_loader, valid_loader, test_image_paths

    def get_pkldata_loader(self, fold, seed=777):
        image_dfs = torch.load(join(self.config.data_path, 'stone_data.pkl'))
        # image_dfs = torch.load(join(self.config.data_path, 'new_stone_data.pkl'))

        for i in range(len(image_dfs)):
            image_text = image_dfs[i]['image_text']
            organ_idx = np.argmax(image_text[0:3])
            x = image_text[4]
            y = image_text[3]
            if organ_idx == 0:
                if x < 0.14 or y < 0.06:
                    organ_idx = 3
            elif organ_idx == 1:
                if x > 0.9 or y < 0.09:
                    organ_idx = 3
            elif organ_idx == 2:
                if x < 0.28 or x > 0.75:
                    organ_idx = 3
            image_dfs[i]['image_text'] = np.append(np.eye(4)[organ_idx], image_text[3:])

        train_index, test_index = self.get_n_fold(image_paths=image_dfs, fold=fold, seed=seed)

        train_image_dfs = [image_dfs[i] for i in train_index]
        test_image_dfs = [image_dfs[i] for i in test_index]

        print("Train images %d: " % (len(train_image_dfs)))
        print("Test images %d: " % (len(test_image_dfs)))

        train_dataset = StoneBiClsPlkDataSet(train_image_dfs, input_size=(self.config.patch_x, self.config.patch_y),
                                             augment=transform_local,num_classes=self.config.classes,
                                             stone_label_smoothing=0.1)
        self.get_Kclass_queue = train_dataset.get_Kclass_queue
        self.getPairInfo = train_dataset.getPairInfo
        valid_dataset = StoneBiClsPlkDataSet(test_image_dfs, input_size=(self.config.patch_x, self.config.patch_y),
                                             augment=transform_test, num_classes=self.config.classes,
                                             stone_label_smoothing=0.1)
        train_loader = DataLoader(train_dataset, batch_size=self.config.train_det_batch_size, shuffle=True,
                                  num_workers=self.config.workers,
                                  collate_fn=self.merge_batch,
                                  drop_last=True if len(train_image_dfs) % self.config.train_det_batch_size != 0 else False)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                                  num_workers=0,
                                  collate_fn=self.merge_batch)
        return train_loader, valid_loader, test_image_dfs

    def get_data_loader(self, fold, seed=777):
        image_paths = self.get_image_full_list(seed=seed)
        train_index, test_index = self.get_n_fold(image_paths=image_paths, fold=fold, seed=seed)


        train_image_paths = [image_paths[i] for i in train_index]
        test_image_paths = [image_paths[i] for i in test_index]
        train_img_names = [basename(p[0]) for p in train_image_paths]
        test_img_names = [basename(p[0]) for p in test_image_paths]

        print("Train images %d: " % (len(train_img_names)), train_img_names)
        print("Test images %d: " % (len(test_img_names)), test_img_names)

        train_dataset = StoneBiClsDataSet(train_image_paths, input_size=(self.config.patch_x, self.config.patch_y),
                                             augment=transform_local, num_classes=self.config.classes)
        self.get_Kclass_queue = train_dataset.get_Kclass_queue
        valid_dataset = StoneBiClsDataSet(test_image_paths, input_size=(self.config.patch_x, self.config.patch_y),
                                             augment=transform_test, num_classes=self.config.classes)
        train_loader = DataLoader(train_dataset, batch_size=self.config.train_det_batch_size, shuffle=True,
                                  num_workers=self.config.workers,
                                  collate_fn=self.merge_batch,
                                  drop_last=True if len(train_image_paths) % self.config.train_det_batch_size != 0 else False)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                                  num_workers=0,
                                  collate_fn=self.merge_batch)
        return train_loader, valid_loader, test_image_paths

    def save_cam(self, logits, image, cam_extractor, file_name):
        activation_map = cam_extractor(logits.squeeze(0).argmax().item(), logits)
        overlay = to_pil_image(activation_map[0].squeeze(0), mode='F').resize(to_pil_image(image.squeeze(0)).size,
                                                                              resample=Image.BICUBIC)
        # plt.imshow(activation_map[0].cpu().squeeze(0).numpy());
        result = overlay_mask(to_pil_image(image.squeeze(0)), to_pil_image(activation_map[0].squeeze(0), mode='F'),
                              alpha=0.7)
        # Display it
        result.save(file_name)
        return np.asarray(overlay, dtype=np.float32)
        # plt.imshow(result);
        # plt.axis('off');
        # plt.tight_layout();
        # # plt.show()
        # plt.savefig(join(self.config.submission_dir, file_name))
        # plt.close()

    def max_norm_cam(self, cam_g, e=1e-5):
        saliency_map = np.maximum(0, cam_g)
        saliency_map_min, saliency_map_max = np.min(saliency_map), np.max(saliency_map)
        cam_out = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min + e)
        cam_out = np.maximum(0, cam_out)
        return cam_out

    def get_queue(self):
        k_class_queue = self.get_Kclass_queue()
        k_class_queue = torch.cat(k_class_queue, dim=0)
        k_class_queue = k_class_queue.to(self.equipment).float()
        return k_class_queue

    def get_pairinfo(self, pred_cls):
        image_pair, image_text_pair = self.getPairInfo(pred_cls)
        # image_pair = torch.unsqueeze(image_pair, dim=0)
        # image_text_pair = torch.tensor(image_text_pair)
        return image_pair.numpy(), image_text_pair, -1
