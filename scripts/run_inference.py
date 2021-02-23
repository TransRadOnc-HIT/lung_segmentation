"Simple script to run segmentation inference"
import os
import argparse
import glob
from lung_segmentation.utils import create_log, untar, get_files
from lung_segmentation.inference import LungSegmentationInference
from lung_segmentation.configuration import STANDARD_CONFIG


def main():

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--input_path', '-i', type=str,
                        help=('This can be either an existing Excel file with a list of all the '
                              'folders containing DICOMS (one subject in each row) or an existing '
                              'directory with one subject in each sub-folder.'))
    PARSER.add_argument('--root_path', '-r', type=str, default='',
                        help=('If an Excel sheet is provided as input, this is the path that '
                              'would be appended to all the folders in each row. Default is empty.'))
    PARSER.add_argument('--work_dir', '-w', type=str,
                        help=('Directory where to store the results.'))
    PARSER.add_argument('--weights', nargs='+', type=str, default=None,
                        help=('Path to the CNN weights to be used for the inference '
                              ' More than one weight can be used, in that case the median '
                              'prediction will be returned. If not provided, they will be '
                              'downloaded from our server'))
    PARSER.add_argument('--evaluate', action='store_true',
                        help=('If ground truth lung masks are available (i.e., the "masks" column was '
                              'provided in the Excel sheet used as input), the result of the '
                              'segmentation can be tested against them. In this case, both '
                              'Dice score and Hausdorff distance will be calculated. '
                              'Default is False.'))

    ARGS = PARSER.parse_args()

#     PARENT_DIR = os.path.abspath(os.path.join(os.path.split(__file__)[0], os.pardir))
    home_dir = os.path.expanduser("~")                                                                                                                                                                 
    PARENT_DIR = os.path.join(home_dir, '.lung_segmentation')
    if not os.path.isdir(PARENT_DIR):
        os.makedirs(PARENT_DIR)
    BIN_DIR = os.path.join(PARENT_DIR, 'bin/')
    WEIGHTS_DIR = os.path.join(PARENT_DIR, 'weights/')
    BIN_URL = 'https://angiogenesis.dkfz.de/oncoexpress/software/delineation/bin/bin.tar.gz'

    CONFIG = STANDARD_CONFIG

# DEEP_CHECK checks some information in the DICOM header of the CTs. This is based on our images
# so if you get errors related to missing DICOM field, just uncomment DEEP_CHECK = False below
# to ignore this check. This can happen because the check is too stringent for your data.
    DEEP_CHECK = CONFIG['dicom_check']
#    DEEP_CHECK = False # uncomment this if you get errors related to some missing DICOM field
    NEW_SPACING = CONFIG['spacing']
    CLUSTER_CORRECTION = CONFIG['cluster_correction']
    WEIGHTS_URL = CONFIG['weights_url']
    MIN_EXTENT = CONFIG['min_extent']

    os.environ['bin_path'] = BIN_DIR

    LOG_DIR = os.path.join(ARGS.work_dir, 'logs')
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)

    LOGGER = create_log(LOG_DIR)

    if ARGS.weights is None and WEIGHTS_URL is not None:
        if not os.path.isdir(WEIGHTS_DIR):
            LOGGER.info('No pre-trained network weights, I will try to download them.')
            try:
                TAR_FILE = get_files(WEIGHTS_URL, PARENT_DIR, 'weights')
                untar(TAR_FILE, 'weights', move_weights=True)
            except:
                LOGGER.error('Pre-trained weights cannot be downloaded. Please check '
                             'your connection and retry or download them manually '
                             'from the repository.')
                raise Exception('Unable to download network weights!')
        else:
            LOGGER.info('Pre-trained network weights found in %s', WEIGHTS_DIR)

        WEIGHTS = [w for w in sorted(glob.glob(os.path.join(WEIGHTS_DIR, '*.h5')))]

        DOWNLOADED = True
    elif ARGS.weights is not None:
        WEIGHTS = ARGS.weights
        DOWNLOADED = False
    else:
        LOGGER.error('If you choose to do not use any configuration file, '
                     'then you must provide the path to the weights to use for '
                     'inference!')
        raise Exception('No weights can be found')
    if len(WEIGHTS) == 5 and DOWNLOADED:
        LOGGER.info('%s weights files found in %s. Five folds inference will be calculated.',
                    len(WEIGHTS), WEIGHTS_DIR)
    elif WEIGHTS and len(WEIGHTS) < 5 and DOWNLOADED:
        LOGGER.warning('Only %s weights files found in %s. There should be 5. Please check '
                       'the repository and download them again in order to run the five folds '
                       'inference will be calculated. The segmentation will still be calculated '
                       'using %s-folds cross validation but the results might be sub-optimal.',
                       len(WEIGHTS), WEIGHTS_DIR, len(WEIGHTS))
    elif len(WEIGHTS) > 5 and DOWNLOADED:
        LOGGER.error('%s weights file found in %s. This is not possible since the model was '
                     'trained using a 5-folds cross validation approach. Please check the '
                     'repository and remove all the unknown weights files.',
                     len(WEIGHTS), WEIGHTS_DIR)
    elif not WEIGHTS:
        LOGGER.error('No weights file found in %s. Probably something went wrong '
                     'during the download. Try to download them directly from %s '
                     'and provide them as input with --weights.', WEIGHTS_DIR, WEIGHTS_URL)
        raise Exception('No weight files found!')

    if not os.path.isdir(BIN_DIR):
        LOGGER.info('No directory containing the binary executables found. '
                    'They will be downloaded from the repository.')
        try:
            TAR_FILE = get_files(BIN_URL, PARENT_DIR, 'bin')
            untar(TAR_FILE, '')
        except:
            LOGGER.error('Binary files cannot be downloaded. Please check '
                         'your connection and retry or download them manually '
                         'from the repository.')
            raise Exception('Unable to download binary files!')
    else:
        LOGGER.info('Binary executables found in %s', BIN_DIR)

    LOGGER.info('The following configuration will be used for the inference:')
    LOGGER.info('Input path: %s', ARGS.input_path)
    LOGGER.info('Working directory: %s', ARGS.work_dir)
    LOGGER.info('Root path: %s', ARGS.root_path)
    LOGGER.info('New spacing: %s', NEW_SPACING)
    LOGGER.info('Weight files: \n%s', '\n'.join([x for x in sorted(WEIGHTS)]))

    INFERENCE = LungSegmentationInference(ARGS.input_path, ARGS.work_dir, deep_check=DEEP_CHECK)
    INFERENCE.get_data(root_path=ARGS.root_path)
    INFERENCE.preprocessing(new_spacing=NEW_SPACING, accurate_naming=False)
    INFERENCE.create_tensors()
    INFERENCE.run_inference(weights=WEIGHTS)
    INFERENCE.save_inference(cluster_correction=CLUSTER_CORRECTION, min_extent=MIN_EXTENT)
    if ARGS.evaluate:
        INFERENCE.run_evaluation()

    print('Done!')


if __name__ == "__main__":

    main()
