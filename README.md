# ICST-SENSORS_Super-Resolution
Super-Resolution method based on Artificial Neural Network and kernel based neighbourhood pixel information, developed to increase spatial resolution of multispectral images.

This method developed at [Vizlab | X-Reality and GeoInformatics Lab](http://vizlab.unisinos.br/), uses a mixed structure of a sequential neural network with the gathering of spatial information in pixel kernels of the input images.

<img src="https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/9039856/9047668/9047670/1570568701-fig-3-source-large.gif" width="500" alt="Neural network learning process">

For a more in-depth understanding of the method, please consider the following works. To cite each of them please consult the [How to cite](#how-to-cite) section:

---
**Improving spatial resolution of LANDSAT spectral bands from a single RGB image using artificial neural network]**

[Ademir Marques Junior<sup>1</sup>](https://www.researchgate.net/profile/Ademir_Junior), [Pedro Rossa<sup>1</sup>](https://www.researchgate.net/profile/Pedro_Rossa), [Rafael Kenji Horota<sup>1</sup>](https://www.researchgate.net/profile/Rafael_Horota), [Diego Brum<sup>1</sup>](https://www.researchgate.net/profile/Diego_Brum), [Eniuce Menezes de Souza<sup>2</sup>](https://www.researchgate.net/profile/Eniuce_Souza), [Alysson Soares Aires<sup>1</sup>](), [Lucas Kupssinskü<sup>1</sup>](https://www.researchgate.net/profile/Lucas_Kupssinskue), , [Maurício Roberto Veronez](https://www.researchgate.net/profile/Mauricio_Veronez)	
[Luiz Gonzaga Junior<sup>1</sup>](https://www.researchgate.net/profile/Luiz_Gonzaga_da_Silveira_Jr),[Caroline Lessio Cazarin<sup>3</sup>](https://www.researchgate.net/profile/Caroline_Cazarin)

[Vizlab | X-Reality and GeoInformatics Lab<sup>1</sup>](http://vizlab.unisinos.br/), 
[Department of Statistics - State University of Maringá - PR, Brazil<sup>2</sup>](http://www.uem.br/international),
[CENPES-Petrobras<sup>3</sup>](https://petrobras.com.br/en/our-activities/technology-innovation/)

To appear in: [2019 13th International Conference on Sensing Technology (ICST)](https://ieeexplore.ieee.org/document/9047670)

**Abstract:**  Spectral information provided by multispectral and hyperspectral sensors has a great impact on remote sensing studies. These sensors are embedded in aircrafts and satellites like the Landsat, which has more data freely available but lack the spatial resolution that suborbital sensors have. To increase the spatial resolution, a series of techniques have been developed like pansharpenning data fusion and more advanced convolutional neural networks for super-resolution, however, the later requires large datasets. To overcome this requirement, this work aims to increase the spatial resolution of Landsat spectral bands using artificial neural networks that uses pixel kernels of a single high-resolution image from Google Earth. Using this method, the high-resolution spectral bands were generated with pixel size of 1m in contrast to the 15m of pansharpenned Landsat bands. The evaluate the predicted spectral bands the validation measures Universal Quality Index (UQI) and Spectral Angle Mapper (SAM) were used, showing values of 0.98 and 0.16 respectively, presenting good results.


---

**Improving spatial resolution of multispectral rock outcrop images using RGB data and artificial neural networks**

Authors: [Ademir Marques Junior<sup>1</sup>](https://www.researchgate.net/profile/Ademir_Junior), [Eniuce Menezes de Souza<sup>2</sup>](https://www.researchgate.net/profile/Eniuce_Souza), [Mariane Müller<sup>1</sup>](https://www.researchgate.net/profile/Marianne_Muller), [Diego Brum<sup>1</sup>](https://www.researchgate.net/profile/Diego_Brum), [Daniel Zanotta<sup>1</sup>](https://www.researchgate.net/profile/Daniel_Zanotta), [Rafael Kenji Horota<sup>1</sup>](https://www.researchgate.net/profile/Rafael_Horota), [Lucas Kupssinskü<sup>1</sup>](https://www.researchgate.net/profile/Lucas_Kupssinskue), [Maurício Roberto Veronez<sup>1</sup>](https://www.researchgate.net/profile/Mauricio_Veronez)	
[Luiz Gonzaga Junior<sup>1</sup>](https://www.researchgate.net/profile/Luiz_Gonzaga_da_Silveira_Jr), [Caroline Lessio Cazarin<sup>3</sup>](https://www.researchgate.net/profile/Caroline_Cazarin)

[Vizlab | X-Reality and GeoInformatics Lab<sup>1</sup>](http://vizlab.unisinos.br/), 
[Department of Statistics - State University of Maringá - PR, Brazil<sup>2</sup>](http://www.uem.br/international),
[CENPES-Petrobras<sup>3</sup>](https://petrobras.com.br/en/our-activities/technology-innovation/)

To appear in: Not yet published

**Abstract:** Spectral information provided by multispectral and hyperspectral sensors has a great impact on remote sensing studies, easing the identification of carbonate outcrops that contribute to a better understanding of petroleum reservoirs. Sensors aboard satellites like Landsat series, which have data freely available usually lack the spatial resolution that suborbital sensors have. Many techniques have been developed to improve spatial resolution through data fusion. However, most of them have serious limitations regarding application and scale. Recently Super-Resolution (SR) convolution neural networks have been tested with encouraging results. However, they require large datasets, more time and computational power for training. To overcome these limitations, this work aims to increase the spatial resolution of multispectral bands from the Landsat satellite database using a modified artificial neural network that uses pixel kernels of a single spatial high-resolution RGB image from Google Earth as input. The methodology was validated with a common dataset of indoor images as well as a specific area of Landsat 8. Different downsized scale inputs were used for training where the validation used the ground truth of the original size images, obtaining comparable results to the recent works. With the method validated, we generated high spatial resolution spectral bands based on RGB images from Google Earth on a carbonated outcrop area, which was then properly classified according to the soil spectral responses making use of the advantage of a higher spatial resolution dataset.

---

# Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Credits](#credits)
- [How to cite](#how-to-cite)


## Requirements

This Python script requires a Python 3 environment and the following installed libraries as seen in the file requirements.txt

- [keras](https://keras.io/)
- [Tensorflow](https://www.tensorflow.org/)
- [Numpy](https://numpy.org/)
- [csv](https://docs.python.org/3/library/csv.html)
- [sewar](https://pypi.org/project/sewar/)
- [Rasterio](https://pypi.org/project/rasterio/)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [Pillow]()

## Installation

For installation download and unpack the content of this repository in a folder acessible by your working Python 3 environment.


## Usage


To replicate and run the validation routine run the scripts bellow in your working folder

```bash
python validation_datasets_routine.py
```

This script will perform the trainning and validation based on downscaled images from the CAVE dataset in order to generate higher resolution images that are then compared to the original size images of the dataset performing then the spectral comparison metrics.

To replicate and run the trainning and validation routine bellow to generate higher spatial resolution near infrared and short-wave infrared bands of Landsat satellite.

```bash
python Landsat_final_superresolution.py
```

To use this methodology in your own dataset import the functions and classes from the scripts bellow:

    from classes import Model

or

    from functions import *
   
   
Use the functions as bellow to import the images and train your model

```bash
x = load_geo_image (lowresolution_RGB_file, remove_mask = False)
y = load_geo_image (lowresolution_MS_file, remove_mask = False)

epochs = 1000
model = Model(x/255, y/255)        
model.modelSetup(epochs)
r2 = model.modelEvaluate()
mse_mae_history = np.stack((history.history['mse'],
                            history.history['mae']), axis = 0).T
```

To generate the higher resolution image provide to the function bellow a high resolution RGB image of the same object used during training.

    new_image = model.predictBands(highresolution_RGB_file/255)

Use Numpy reshape to format the output image to the same shape as the High resolution RGB image.


## Credits	
This work is credited to the [Vizlab | X-Reality and GeoInformatics Lab](http://vizlab.unisinos.br/) and the following authors and developers:	
Author Name  | Profile	
------------- | -------------	
Ademir Marques Junior | https://www.researchgate.net/profile/Ademir_Junior	
Eniuce Menezes de Souza | https://www.researchgate.net/profile/Eniuce_Souza	
Mariane Müller | https://www.researchgate.net/profile/Marianne_Muller	
Diego Brum | https://www.researchgate.net/profile/Diego_Brum	
Daniel Zanotta | https://www.researchgate.net/profile/Daniel_Zanotta	
Rafael Kenji Horota | https://www.researchgate.net/profile/Rafael_Horota	
Lucas Kupssinskü | https://www.researchgate.net/profile/Lucas_Kupssinskue	
Pedro Rossa | https://www.researchgate.net/profile/Pedro_Rossa	
Maurício Roberto Veronez | https://www.researchgate.net/profile/Mauricio_Veronez	
Luiz Gonzaga Junior | https://www.researchgate.net/profile/Luiz_Gonzaga_da_Silveira_Jr	
Caroline Lessio Cazarin | https://www.researchgate.net/profile/Caroline_Cazarin


## License

GNU GP3. See LICENSE for full details. 

## How to cite

If you find our work useful in your research please consider citing one of our papers:

```bash
@inproceedings{marques2019improving,
  title={Improving spatial resolution of LANDSAT spectral bands from a single RGB image using artificial neural network},
  author={Marques, Ademir and Rossa, Pedro and Horota, Rafael Kenji and Brum, Diego and de Souza, Eniuce Menezes and Aires, Alyson Soares and Kupssinsk{\"u}, Lucas and Veronez, Maur{\'\i}cio Roberto and Gonzaga, Luis and Cazarin, Caroline Lessio},
  booktitle={2019 13th International Conference on Sensing Technology (ICST)},
  pages={1--6},
  year={2019},
  organization={IEEE}
}
```







