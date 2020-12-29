.PHONY: clean data

########################################################################
# Get data
########################################################################

data: social_network hotel_review

social_network:
	python3 src/data/make_dataset.py kaggle-social-network -f data/raw/

hotel_review:
	python3 src/data/make_dataset.py kaggle-tripadvisor-hotel-reviews -f data/raw/
