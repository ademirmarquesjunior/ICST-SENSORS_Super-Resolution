from imports import *


def rmse(original, predicted):
    '''
    

    Parameters
    ----------
    original : TYPE
        DESCRIPTION.
    predicted : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    try:
        bands = np.shape(original)[2]
    except:
        bands = 1
    
    if bands >1:
        np.shape(original)[2]
        mse = 0
        for i in range(0,np.shape(original)[2]):
            err = np.sum((original[:,:,i].astype("float") - predicted[:,:,i].astype("float")) ** 2)
            err /= float(original.shape[0] * original.shape[1])
            mse = mse + err
        mse = mse / np.shape(original)[2]
        rmse = math.sqrt(mse)
    else:
        mse = np.sum((original.astype("float") - predicted.astype("float")) ** 2)
        mse /= float(original.shape[0] * original.shape[1])
        rmse = math.sqrt(mse)

    return rmse



def ergas(original, predicted, x):
    '''
    ERGASS quality index function (redone due to the Sewar package equivalent
    function was not returning accurate values)
    Uses Sewar RMSE function

    Parameters
    ----------
    original : Array
        Numpy multiband image array.
    predicted : Array
        Numpy multiband image array.
    x : Array
        DESCRIPTION.

    Returns
    -------
    ergas: float
        ERGASS quality index value between two multiband images.

    '''
    
    try:
        K = np.shape(original)[2]
    except:
        K = 1
            
    aux = 0

    for i in range(0,K):
        aux += math.pow(rmse(original, predicted)/np.mean(original),2)
        if K > 1:
            aux += math.pow(
                rmse(original[:,:,i], predicted[:,:,i])
                /np.mean(original[:,:,i]),2)
            
    ergas = 100*((np.shape(x)[0]*
                  np.shape(x)[0])/(np.shape(original)[0]*
                                   np.shape(original)[0]))*math.sqrt((1/K)*aux) 
        
    return ergas
       


def concatenate_images (folder, element, bands, filetype):
    '''
    Concatenate multiple image bands to a single multidimensional array,
    The pattern of the images must be "element_ms_##.filetype". 
    ## means two digit number

    Parameters
    ----------
    folder : str
        Folder path of the working images.
    element : str
        Key name of the files to load.
    bands : int
        Quantity of bands (individual grayscale images) to load.
    filetype : str
        DESCRIPTION.

    Returns
    -------
    output_file : Array
        Multiband Numpy array.

    '''
    
    output_file = []
    filename  = element+"_ms_" #"balloons_ms_"
    filetype = "." + filetype
    for i in range(1, bands+1):
        band = str(i)
        if i < 10: band = "0" + band
        aux = cv2.imread(folder + filename + band + filetype)[:,:,0:1]
        print("Loading: " + filename + band + filetype)
        if np.size(output_file) == 0:
            output_file = aux
        else:
            output_file = np.concatenate((output_file, aux), axis=2)
            
    return output_file



def load_geo_image (filename, remove_mask):
    '''
    Load tif image to a Numpy array

    Parameters
    ----------
    filename : str
        Name file with address to open.
    remove_mask : bool
        Considers only the 3 band/layers of the file.

    Returns
    -------
    geo_image : array
        Numpy array of the bands/layers loaded.

    '''
    geo_image = rt.open(filename)
    geo_image = geo_image.read()
    geo_image = np.moveaxis(geo_image, 0,-1)
    if remove_mask == True: geo_image = geo_image[:,:,0:3]
    return geo_image

#TODO: Correct affine transformation
def save_geo_image (array, profile, filename):
    '''
    Save Numpy array of image/composite spectral bands.

    Parameters
    ----------
    array : Numpy array
        Numpy array of image/composite spectral bands.
    profile : Rasterio/Gdal profile
        Rasterio/Gdal profile.
    filename : str
        Save filename.

    Returns
    -------
    None.

    '''
    
    profile["width"] = np.shape(array)[1]
    profile["height"] = np.shape(array)[0]
    
    try:
        profile["count"] = np.shape(array)[2]
        array = np.moveaxis(array, 2,0)
    except:
        profile["count"] = 1
        array = np.reshape(array, (1, np.shape(array)[0],
                                                     np.shape(array)[1]))
    
    with rt.open(filename,
                 'w', **profile) as dst:
        dst.write(array)
        
    return
    
   

'''
Plots


'''


def plot_mse_mae_history(element, ratio, epochs, csv_file, folder):
    '''
    Plot MSE and MAE of history training.

    Parameters
    ----------
    element : str
        Key name of the work element.
    ratio : TYPE
        Relation between 
    epochs : int
        Number of epochs during training.
    csv_file : str
        Address of the CSV to be loaded.
    folder : str
        Folder address to save the plot.

    Returns
    -------
    None.

    '''
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        headers = next(reader)
        mse_mae_history = np.array(list(reader)).astype(float)
    
    #Print Trainning graph
    plt.figure()
    plt.plot(mse_mae_history[:,0], label = 'MSE')
    plt.plot(mse_mae_history[:,1], label = 'MAE')
    plt.title(element + ' - Trainning metrics for ' + str(epochs) + 
              ' epochs - ' + str(ratio) + ' Ratio\n MSE final: ' + 
              str(mse_mae_history[:,0][-1]) + ' - MAE final: ' +
              str(mse_mae_history[:,1][-1]))
    plt.ylim(0, 0.2)
    plt.legend()
    plt.savefig(folder + element + '_ratio_' + str(ratio) + '_trainning.png')
    plt.close()
    
    return


def plot_R2 (element, epochs, csv_file, folder, labels):
    '''
    Plot R2 statistics

    Parameters
    ----------
    element : str
        Key name of the work element.
    epochs : int
        Number of epochs during training.
    csv_file : str
        Address of the CSV to be loaded.
    folder : str
        Folder address to save the plot.
    labels : list
        List of labels.

    Returns
    -------
    None.

    '''

    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        headers = next(reader)
        r2_results = np.array(list(reader)).astype(float) 

    plt.figure('r2')
    for i in range(0,np.shape(r2_results)[0]):
        plt.plot(r2_results[i], label = labels[i])

    plt.xlabel('bands')
    plt.xticks(np.arange(3), ["5", "6", "7"])
    plt.legend()
    plt.title(element + ' - R2 metric after ' + str(epochs) + ' epochs')
    plt.savefig(folder + element + '_R2_metric.png')
    plt.close('r2')
    
    return


def plot_image_diff (image_1, image_2, ratio, band_number, save_file_address):
    '''
    Plot and calculate descriptive stattistics based on the image differences.

    Parameters
    ----------
    image_1 : array
        Grayscale image (2 dimensional array).
    image_2 : array
        Grayscale image (2 dimensional array).
    ratio : int
        Number id referencing the band or layer of the image.
    band_number : int
        Number id referencing the band or layer of the image.
    save_file_address : str
        Name and address to save the plot.

    Returns
    -------
    float
        Mean of all the differences between the grayscale images.
    int
        Median of all the differences between the grayscale images.
    float
        Standard deviation of all the differences between the grayscale images.

    '''
     
    diff = ImageChops.difference(image_1, image_2)
    
    #Print and save the image difference with scale range
    plt.figure(band_number)
    ax = plt.subplot(111)
    im = ax.imshow(diff)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig(save_file_address)
    plt.close(band_number) 

    #Calculate descriptive statistics based on the image difference
    stat = ImageStat.Stat(diff)

    return stat.mean[0], stat.median[0], stat.stddev[0]


