from main import train
import nni
import logging
import argparse


def get_params():
    parser = argparse.ArgumentParser(description='RS_Segmentation_PyTorch')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    '''
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        logger.debug(params)
        train(params)
    except Exception as exception:
        logger.exception(exception)
        raise
    '''
    params = vars(get_params())
    train(params)
