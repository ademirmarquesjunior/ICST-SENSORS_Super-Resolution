from imports import *
from functions import *
from classes import Model

wd = os.getcwd()
elements = ("beads", "balloons", "clay", "cloth", "flowers", "glass_tiles", 
            "Landsat")

for element in elements:
    #element = elements[-1]
    folder = wd + "\\datasets\\" + element + "\\"

    if element == "Landsat":
        rgb = load_geo_image(folder + element + "_RGB.tif", True)
        ms_original = load_geo_image(folder + element + "_ms.tif", True)
    else:   
        rgb = load_geo_image(folder + element + "_RGB.bmp", False)    
        ms_original = concatenate_images (folder, element, 31, "png")
       
    quality_individual_bands = []
    r2_results = []
    quality_composed_image = []
    
    for ratio in (2, 4, 8, 16, 32):
        #ratio = 2
        
        #Load and resize trainning images according desired ratio
        x = cv2.resize(rgb, (int(np.shape(rgb)[1]/ratio),
                             int(np.shape(rgb)[0]/ratio)))
        
        y = cv2.resize(ms_original, (int(np.shape(rgb)[1]/ratio),
                                     int(np.shape(rgb)[0]/ratio)))
        
        x_resized = cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2RGB), 
                          (np.shape(rgb)[1], np.shape(rgb)[0]), 
                          interpolation=0)
        y_resized = cv2.resize(y, (np.shape(ms_original)[1],
                                          np.shape(ms_original)[0]),
                               interpolation=0)
        
        
        cv2.imwrite(folder + element + '_resized_RGB_' + str(ratio) 
                    + '_ratio.png', cv2.cvtColor(x_resized, cv2.COLOR_BGR2RGB))
        
        cv2.imwrite(folder + element + '_resized_MS_' + str(ratio) 
                    + '_ratio.png', cv2.cvtColor(y_resized, cv2.COLOR_BGR2RGB)) #y[:,:,15]

        #Initialize model
        model = Model(x/255, y/255)        
        model.modelSetup()
        
        #Train model
        epochs = 100
        history = model.modelTrainnning(epochs)
        
        #Save history training
        mse_mae_history = np.stack((history.history['mse'],
                                    history.history['mae']), axis = 0).T
        
        csv_file = folder + element + '_ratio_' + str(ratio) + '_mse_mae.csv'
        np.savetxt(csv_file, mse_mae_history, fmt='%3.7f', 
                   delimiter=';', header='MSE;MAE')
        
        plot_mse_mae_history(element, ratio, epochs, csv_file, folder)
        
        #Evaluate the model using the validation set
        r2 = model.modelEvaluate()
        if np.size(r2_results) == 0:
            r2_results = r2
        else:
            r2_results = np.vstack((r2_results,r2))
        
        #Generate the new image using the trainned model
        #Image.fromarray(rgb).show()
        new_image = model.predictBands(rgb/255)
        
        #Reshape predicted to the same shape as multispectral desired image
        ms_predicted_0 = np.reshape(new_image, (np.shape(ms_original)[0]-2, 
                                                np.shape(ms_original)[1]-2, 
                                                np.shape(ms_original)[2]))
        
        #Convert to 8bit color values
        ms_predicted_0 = np.uint8(ms_predicted_0*255)
        
        #Pad the predicted to the same size as the original spectral image  
        ms_predicted = np.pad(ms_predicted_0,((0,2),(0,2),(0,0)), mode="edge")
        
        #For Geotiff images copy the profile and save to a georeferenced file
        if element == "Landsat":
            profile = rt.open(folder + element + "_RGB.tif").profile
            filename = folder + 'composed_Landsat_ms.tif'
            save_geo_image (ms_predicted, profile, filename)
            
       
        #Generate statistics for composed multispectral images
        quality_composed_image.append(sw.sam(ms_original, ms_predicted))
        quality_composed_image.append(sw.rmse(ms_original, ms_predicted))
        quality_composed_image.append(sw.rmse(ms_original/255,
                                              ms_predicted/255))
        quality_composed_image.append(ergas(ms_original, ms_predicted, x))
        quality_composed_image.append(sw.uqi(ms_original, ms_predicted))
    
    
        #Generate statistics for each band
        results = []
        for band_number in range(0,np.shape(ms_original)[2]):
            #Load individual slices of the images to 2D arrays
            predicted = ms_predicted[:,:,band_number]
            original = ms_original[:,:,band_number]
            
            plot_file = folder + element + '_ratio_' + str(
                ratio) + '_band_' + str(band_number)+'.png'
            #Calculate original and predicted image differences
            stat = plot_image_diff(Image.fromarray(predicted),
                                   Image.fromarray(original),
                                   ratio, band_number, plot_file)
            
            results.append(stat[0]) #Mean
            results.append(stat[1]) #Median
            results.append(stat[2]) #Standar deviation
                
            #Save the generated band individually
            if element == "Landsat":
                profile = rt.open(folder + element + "_RGB.tif").profile
                filename = folder + 'generated_' + str(
                    element) + '_ratio_' + str(
                        ratio) + '_band_' + str(band_number)+'.tif'
                save_geo_image (predicted, profile, filename)                    
            else:
                cv2.imwrite('generated_' + str(element) + '_ratio_' + str(
                    ratio) + '_band_' + str(band_number)+'.png', predicted)
    
            #Compute quality metrics for image comparison
            results.append(sw.sam(original, predicted)) #Spectral Angle Mapper
            results.append(sw.rmse(original, predicted)) #RMSE
            results.append(sw.rmse(original/255, predicted/255)) #RMSE norm
            results.append(ergas(ms_original, ms_predicted, x)) #Ergas
            results.append(sw.uqi(original, predicted)) #UQI
        
        #Save each band quality metric results to CSV for the given ratio
        results = np.reshape(results, (np.shape(ms_original)[2],8))
        header = 'mean; median; std; sam; rmse; rmse; ergas; uqi'
        np.savetxt(element + '_quality_results_ratio_'+ str(
            ratio) +'.csv',results, fmt='%3.5f', delimiter=';', header=header)
        
        #Append each band quality metrics for each ratio in a single array
        #to turn easier to plot graphs between different metrics and ratio
        if np.size(quality_individual_bands) == 0:
            quality_individual_bands = results
        else:
            quality_individual_bands = np.dstack((
                quality_individual_bands, results))   
     
    #Save quality metrics for the composed multispectral products
    header = 'sam; rmse; rmse; ergas; uqi'
    np.savetxt(folder + element + '_composed_quality_results.csv', np.reshape(
        quality_composed_image, (5,5)), delimiter=';', fmt='%3.5f',
        header=header)  
    
    #Save R2 correlation metrics
    header = ""
    for  band_number in range(0,np.shape(ms_original)[2]):
        header+=str(band_number) + ";"
    np.savetxt(folder + element + '_r2_results.csv', r2_results,
               delimiter=';', fmt='%3.5f', header=header)
    
    #Load and plot R2 metrics
    csv_file = folder + element + '_r2_results.csv'
    labels = ['ratio 2', 'ratio 4', 'ratio 8', 'ratio 16', 'ratio 32']
    plot_R2 (element, epochs, csv_file, folder, labels)
    
 