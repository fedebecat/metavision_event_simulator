#CRF RGB 
python create_dataset.py /equilibrium/datasets/cityscapes/gtFine --quality 10 --dst-path /home/nniccoli/cv/datasets --rgb
python create_dataset.py /equilibrium/datasets/cityscapes/gtFine --quality 25 --dst-path /home/nniccoli/cv/datasets --rgb
python create_dataset.py /equilibrium/datasets/cityscapes/gtFine --quality 35 --dst-path /home/nniccoli/cv/datasets --rgb
python create_dataset.py /equilibrium/datasets/cityscapes/gtFine --quality 45 --dst-path /home/nniccoli/cv/datasets --rgb

#JPEG RGB
python create_dataset.py /equilibrium/datasets/cityscapes/gtFine --quality 100 --dst-path /home/nniccoli/cv/datasets --jpeg-compression --rgb
python create_dataset.py /equilibrium/datasets/cityscapes/gtFine --quality 25 --dst-path /home/nniccoli/cv/datasets --jpeg-compression --rgb
python create_dataset.py /equilibrium/datasets/cityscapes/gtFine --quality 50 --dst-path /home/nniccoli/cv/datasets --jpeg-compression --rgb

#CRF eventi 
python create_dataset.py /equilibrium/datasets/cityscapes_event/gtFine --quality 10 --dst-path /home/nniccoli/cv/datasets
python create_dataset.py /equilibrium/datasets/cityscapes_event/gtFine --quality 25 --dst-path /home/nniccoli/cv/datasets
python create_dataset.py /equilibrium/datasets/cityscapes_event/gtFine --quality 35 --dst-path /home/nniccoli/cv/datasets
python create_dataset.py /equilibrium/datasets/cityscapes_event/gtFine --quality 45 --dst-path /home/nniccoli/cv/datasets

#JPEG eventi
python create_dataset.py /equilibrium/datasets/cityscapes_event/gtFine --quality 100 --dst-path /home/nniccoli/cv/datasets --jpeg-compression
python create_dataset.py /equilibrium/datasets/cityscapes_event/gtFine --quality 25 --dst-path /home/nniccoli/cv/datasets --jpeg-compression
python create_dataset.py /equilibrium/datasets/cityscapes_event/gtFine --quality 50 --dst-path /home/nniccoli/cv/datasets --jpeg-compression


