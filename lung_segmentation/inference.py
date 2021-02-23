"Class to run lung segmentation inference"
import logging
import os
from pathlib import Path
import nrrd
import pickle
import numpy as np
from skimage.transform import resize
from lung_segmentation.utils import (binarization, dice_calculation,
                                     violin_box_plot, run_cluster_correction,
                                     run_hd, batch_processing, resize_image)
from lung_segmentation.models import unet_lung
from lung_segmentation.base import LungSegmentationBase


LOGGER = logging.getLogger('lungs_segmentation')


class LungSegmentationInference(LungSegmentationBase):
    "Class to run the lung segmentation inference and evaluation."
    def get_data(self, root_path=''):
        "Function to get the data for the prediction"
        self.testing = True
        self.predicted_images = []
        if (os.path.isdir(os.path.join(self.work_dir, 'inference'))
                and os.path.isfile(os.path.join(self.work_dir, 'inference', 'test_subjects.txt'))
                and os.path.isfile(os.path.join(self.work_dir, 'inference',
                                                'test_subjects_gt_masks.txt'))):
            with open(os.path.join(self.work_dir, 'inference', 'test_subjects.txt'), 'r') as f:
                self.dcm_folders = [x.strip() for x in f]
            with open(os.path.join(self.work_dir, 'inference',
                                   'test_subjects_gt_masks.txt'), 'r') as f:
                self.mask_paths = [x.strip() for x in f]
        elif os.path.isfile(self.input_path):
            self.dcm_folders, self.mask_paths = batch_processing(self.input_path, root=root_path)
        else:
            LOGGER.info('No folder called "inference" in the working directory.'
                        ' The pipeline will look for DICOM file to use for '
                        'inference in all the sub-folders within the '
                        'working directory.')
            input_dir = Path(self.input_path)
            LOGGER.info('Input directory: {}'.format(input_dir))
            self.dcm_folders = sorted([str(input_dir/x) for x in input_dir.iterdir() if x.is_dir() and
                                      ((input_dir/x).glob('*.dcm') or (input_dir/x).glob('*.DCM')
                                       or (input_dir/x).glob('*.IMA'))])
            LOGGER.info('Found {0} sub-folders in {1}. They will be used to run the inference.'
                        .format(len(self.dcm_folders), str(input_dir)))

        self.work_dir = os.path.join(str(self.work_dir), 'inference')

    def create_tensors(self, patch_size=(96, 96), save2npy=False):
        "Function to create the tensors for the prediction"
        return LungSegmentationBase.create_tensors(self, patch_size=patch_size, save2npy=save2npy)

    def run_inference(self, weights):
        "Function to run the CNN inference"
        test_set = np.asarray(self.image_tensor)
        predictions = []
        LOGGER.info('Segmentation inference started.')
        model = unet_lung()
        for i, weight in enumerate(weights):
            LOGGER.info('Segmentation inference fold {}.'.format(i+1))
            model.load_weights(weight)
            predictions.append(model.predict(test_set))

        predictions = np.asarray(predictions, dtype=np.float16)
        self.prediction = np.mean(predictions, axis=0)

    def save_inference(self, min_extent=10000, cluster_correction=True):
        "Function to save the segmented masks"
        prediction = self.prediction
        if cluster_correction:
            binarize = False
        else:
            binarize = True
        z0 = 0
        for i, image in enumerate(self.image_info):
            try:
                patches = self.image_info[image]['patches']
                slices = self.image_info[image]['slices']
                resampled_image_dim = self.image_info[image]['image_dim']
                indexes = self.image_info[image]['indexes']
                deltas = self.image_info[image]['deltas']
                original_image_dim = self.image_info[image]['orig_size']
                im = prediction[z0:z0+(slices*patches), :, :, 0]
                final_prediction = self.inference_reshaping(
                    im, patches, slices, resampled_image_dim, indexes, deltas,
                    original_image_dim, binarize=binarize)
                outname = image.split('_resampled')[0]+'_lung_segmented.nrrd'
                reference = image.split('_resampled')[0]+'.nrrd'
                _, hd = nrrd.read(reference)
                nrrd.write(outname, final_prediction, header=hd)
                if cluster_correction:
                    outname = run_cluster_correction(outname, 0.2, min_extent)
                self.predicted_images.append(outname)
                z0 = z0+(slices*patches)
            except:
                continue

    @staticmethod
    def inference_reshaping(generated_images, patches, slices,
                            dims, indexes, deltas, original_size,
                            binarize=False):
        "Function to reshape the predictions"
        if patches > 1:
            sl = 0
            final_image = np.zeros((slices, dims[0], dims[1], patches),
                                   dtype=np.float32)-2
            for n in range(0, generated_images.shape[0], patches):
                k = 0
                for j in indexes[1]:
                    for i in indexes[0]:
                        final_image[sl, i[0]:i[1], j[0]:j[1], k] = (
                            generated_images[n+k, deltas[0]:, deltas[1]:])
                        k += 1
                sl = sl + 1
            final_image[final_image==-2] = np.nan
            final_image = np.nanmean(final_image, axis=-1)
            final_image[np.isnan(final_image)] = 0
        else:
            final_image = generated_images[:, deltas[0]:, deltas[1]:]

        final_image = np.swapaxes(final_image, 0, 2)
        final_image = np.swapaxes(final_image, 0, 1)
        if final_image.shape != original_size:
            final_image = resize(final_image.astype(np.float64), original_size, order=0,
                                 mode='edge', cval=0, anti_aliasing=False)
        if binarize:
            final_image = binarization(final_image)

        return final_image

    def run_evaluation(self):
        "Function to evaluate the segmentation w.r.t. a ground truth"
#         for mask in self.preprocessed_masks:
#             bp, name = os.path.split(mask)
#             im_name = 'Raw_data_for_'+name.split('.nrrd')[0]+'_lung_segmented.nrrd'
#             self.predicted_images.append(os.path.join(bp, im_name))
        assert len(self.predicted_images) == len(self.preprocessed_masks)
        all_dsc = []
        all_hd = []
        all_hd_100 = []
        evaluated = []
        for i, predicted in enumerate(self.predicted_images):
            gt = self.preprocessed_masks[i]
            dsc = dice_calculation(gt, predicted)
            hd_95 = run_hd(gt, predicted, mode='95')
            hd_100 = run_hd(gt, predicted)
            if dsc is not None and hd_95 is not None and hd_100 is not None:
                all_hd.append(hd_95)
                all_hd_100.append(hd_100)
                all_dsc.append(dsc)
                evaluated.append(predicted)
            else:
                LOGGER.info('Evaluation metrics cannot be calculated for image {}.'
                            ' Probably the reference and the predicted image have '
                            'different origin. This image will be ignored. '
                            'Please check.'.format(predicted))

        violin_box_plot(all_dsc, os.path.join(self.work_dir, 'DSC_violin_plot.png'))
        violin_box_plot(all_hd, os.path.join(self.work_dir, 'HD_95_violin_plot.png'))
        violin_box_plot(all_hd_100, os.path.join(self.work_dir, 'HD_max_violin_plot.png'))

        LOGGER.info('Median DSC: %s', np.median(all_dsc))
        LOGGER.info('DSC 25th percentile: %s', np.percentile(all_dsc, 25))
        LOGGER.info('DSC 75th percentile: %s', np.percentile(all_dsc, 75))
        LOGGER.info('Median HD_95: %s', np.median(all_hd))
        LOGGER.info('HD_95 25th percentile: %s', np.percentile(all_hd, 25))
        LOGGER.info('HD_95 75th percentile: %s', np.percentile(all_hd, 75))
        LOGGER.info('Median HD_max: %s', np.median(all_hd_100))
        LOGGER.info('HD_max 25th percentile: %s', np.percentile(all_hd_100, 25))
        LOGGER.info('HD_max 75th percentile: %s', np.percentile(all_hd_100, 75))
        LOGGER.info('Max DSC: %s', np.max(all_dsc))
        LOGGER.info('Min DSC: %f', np.min(all_dsc))
        LOGGER.info('Max HD_95: %f', np.max(all_hd))
        LOGGER.info('Min HD_95: %f', np.min(all_hd))
        LOGGER.info('Max HD_max: %f', np.max(all_hd_100))
        LOGGER.info('Min HD_max: %f', np.min(all_hd_100))
        LOGGER.info('Image with minumum DSC: %s',
                    evaluated[np.where(np.asarray(all_dsc)==np.min(all_dsc))[0][0]])
        LOGGER.info('Image with maximum HD_95: %s',
                    evaluated[np.where(np.asarray(all_hd)==np.max(all_hd))[0][0]])
        LOGGER.info('Image with maximum HD_max: %s',
                    evaluated[np.where(np.asarray(all_hd_100)==np.max(all_hd_100))[0][0]])


class IndividualInference(LungSegmentationInference):

    def get_data(self):
        self.testing = True
        self.predicted_images = []

    def preprocessing(self, new_spacing=(0.35, 0.35, 0.35)):

        image = self.input_path
        LOGGER.info('Processing file {}'.format(image))
        if os.path.isfile(os.path.join(self.work_dir, 'image_info.p')):
            with open(os.path.join(self.work_dir, 'image_info.p'), 'rb') as fp:
                self.image_info = pickle.load(fp)

        if new_spacing is not None:
            LOGGER.info('Input image will be now resampled to have '
                        '{0}x{1}x{2} mm resolution.'
                        .format(new_spacing[0], new_spacing[1], new_spacing[2]))
            _, _, img_path, orig_size = resize_image(image, new_spacing=new_spacing)
        else:
            img_path = image
            orig_size = None
        self.preprocessed_images.append(img_path)
        self.image_info[img_path] = {}
        self.image_info[img_path]['orig_size'] = orig_size

        with open(os.path.join(self.work_dir, 'image_info.p'), 'wb') as fp:
            pickle.dump(self.image_info, fp, protocol=pickle.HIGHEST_PROTOCOL)
