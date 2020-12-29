'''
Get data for the tasks

Usage:
    make_dataset.py kaggle-social-network -f OUTPUT_DIR
    make_dataset.py kaggle-tripadvisor-hotel-reviews -f OUTPUT_DIR

Options:
    -f OUTPUT_DIR          Output directory of data

'''


from docopt import docopt
import logging
from kaggle.api.kaggle_api_extended import KaggleApi

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

api = KaggleApi()


def get_social_network_data(opts):
    logger.info('Getting social network data')
    api.dataset_download_files('simonburton/wikipedia-people-network',
                               path=opts['-f'],
                               unzip=True)


def get_hotel_review_data(opts):
    logger.info('Getting hotel review data')
    api.dataset_download_files('andrewmvd/trip-advisor-hotel-reviews',
                               path=opts['-f'],
                               unzip=True)


if __name__ == '__main__':
    opts = docopt(__doc__)
    print(opts)

    logger.info('Authentication with Kaggle ...')
    api.authenticate()
    logger.info('Authentication successful')
    
    if opts['kaggle-social-network']:
        get_social_network_data(opts)
    elif opts['kaggle-tripadvisor-hotel-reviews']:
        get_hotel_review_data(opts)
    else:
        raise NotImplementedError



 
