from pathlib import Path

class PathDict(object):
    @staticmethod
    def get_root_and_output_dirs(dataset: str):
        if dataset == 'ucf101':
            # folder that contains class labels
            root_dir = '/Path/to/UCF-101'

            # Save preprocess data into output_dir
            output_dir = '/path/to/VAR/ucf101'

            return root_dir, output_dir
        elif dataset == 'hmdb51':
            # folder that contains class labels
            root_dir = './dataloaders/hmdb51'

            output_dir = './dataloaders/hmdb51_processed'

            return root_dir, output_dir
        elif dataset == 'kaggle':
            # folder that contains class labels
            root_dir = '../Downloads/deepfake-detection-challenge/train'

            output_dir = './dataloaders/deepfake-processed'

            return root_dir, output_dir
        elif dataset == 'celeb-df':
            # folder that contains class labels
            root_dir = '/data/hayley-old/Github/Reverse_Engineering_GMs/data/originals/Celeb-DF-v2'
            output_dir = '/data/hayley-old/Github/Reverse_Engineering_GMs/data/originals/Celeb-DF-processed' \

            return Path(root_dir), Path(output_dir)
        else:
            print('Database {} not available.'.format(dataset))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return './c3d-pretrained.pth'