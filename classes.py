import functions
from imports import *

#------------------------------------------------------------------#
# Classe de rede neural para predição de dados espectrais de imagens. Utiliza 
# bibliotecas Keras, Numpy e Pillow
class Model():
    
    #
    # Initialize the model and set the dataset for trainning and validation. Both input images must have the same size
    # Args:
    # input # RGB image compose (3 channels)
    # target # Spectral image compose (3 channels)
    def __init__(self, x, y):
        
        self.input_dim = np.shape(x)[2]
        self.output_dim = np.shape(y)[2]

        X = []
        Y = []
        for i in range(1, np.shape(x)[0]-1):
            for j in range(1, np.shape(x)[1]-1):
                aux = []
                for k in range(0, np.shape(x)[2]):
                    aux.append([x[i-1][j-1][k],x[i-1][j][k],x[i-1][j+1][k],x[i][j-1][k],x[i][j][k],x[i][j+1][k],x[i+1][j-1][k],x[i+1][j][k],x[i+1][j+1][k]])
                X.append(np.reshape(aux,(1, np.size(aux))))    
                    
                #X.append([x[i-1][j-1][0],x[i-1][j][0],x[i-1][j+1][0],x[i][j-1][0],x[i][j][0],x[i][j+1][0],x[i+1][j-1][0],x[i+1][j][0],x[i+1][j+1][0],
                #          x[i-1][j-1][1],x[i-1][j][1],x[i-1][j+1][1],x[i][j-1][1],x[i][j][1],x[i][j+1][1],x[i+1][j-1][1],x[i+1][j][1],x[i+1][j+1][1],
                #          x[i-1][j-1][2],x[i-1][j][2],x[i-1][j+1][2],x[i][j-1][2],x[i][j][2],x[i][j+1][2],x[i+1][j-1][2],x[i+1][j][2],x[i+1][j+1][2]])
                
                aux = []
                for k in range(0, np.shape(y)[2]):
                    aux.append([y[i-1][j-1][k]])
                Y.append(np.reshape(aux,(1, np.size(aux))))  
                
                #Y.append([y[i][j][0],y[i][j][1],y[i][j][2]])
                
        X = np.reshape(X, (np.shape(X)[0],np.shape(x)[2]*9))
        Y = np.reshape(Y, (np.shape(Y)[0],np.shape(y)[2]))
        
        #X = X/255
        #Y = Y/255
        
        self.X = X
        self.Y = Y
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
        
    #
    # Initialize the Neural network based on Keras 
    # Args: # To be configured by arguments in future iterations of the code
    #
    def modelSetup(self):
        self.model = Sequential()
        self.model.add(Dense(activation = "relu", input_dim = self.input_dim*9, units = 150, kernel_initializer = "random_uniform", ))
        self.model.add(Dense(activation = "relu", units = 70, kernel_initializer = "random_uniform"))
        self.model.add(Dense(activation = "relu", units = 35, kernel_initializer = "random_uniform"))
        self.model.add(Dense(activation = "sigmoid", units = self.output_dim, kernel_initializer = "random_uniform"))
        self.model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse','mae','acc'])
        #self.tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        return 1
    
    #
    # Start the model trainning
    # Args: # To be configured by arguments in future iterations of the code
    #    
    def modelTrainnning(self, epochs):
        self.history = History()
        self.model.fit(self.X_train, self.Y_train,
        #batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        #shuffle=True,
        #validation_split=0.3,
        callbacks=[self.history]
        )
        return self.history
    
    #
    # Evaluate the network using the validation set and the R2 score
    # Args: no args
    #
    def modelEvaluate(self):
        Y_pred = self.model.predict(self.X_test, verbose = 1)
        r2_results = []
        for i in range(0, np.shape(Y_pred)[1]):
            r2_results.append(r2_score(self.Y_test[:,i],Y_pred[:,i]))

        return np.reshape(r2_results,(1,np.size(r2_results)))
    
    #
    # Save the trainned network to files
    # Args: 
    # json_file - File name for the network model terminated in ".json"
    # weigth_h5_file - File name for network model weights terminated in ".h5" 
    #
    def modelSave(self, json_file, weight_h5_file):
        model_json = model.to_json()
        with open(json_file, "w") as json_file:
            json_file.write(model_json)
        model.save_weights(weight_h5_file)
        return
    
    #
    # Load the trainned network from files
    # Args: 
    # json_file - File name for the network model terminated in ".json"
    # weigth_h5_file - File name for network model weights terminated in ".h5" 
    #    
    def modelLoad(self, json_file, weight_h5_file):
        json_file = open(json_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = Sequential.model_from_json(loaded_model_json)
        model.load_weights(weight_h5_file)
        return        
    
    #
    # Generate a new image based on the trainned network. Returns and save the generate image to a file
    # Args: 
    # image - RGB image in wich we want to predict spectral bands
    #
    def predictBands(self, x):
        X = []
        for i in range(1, np.shape(x)[0]-1):
            for j in range(1, np.shape(x)[1]-1):
                aux = []
                for k in range(0, np.shape(x)[2]):
                    aux.append([x[i-1][j-1][k],x[i-1][j][k],x[i-1][j+1][k],x[i][j-1][k],x[i][j][k],x[i][j+1][k],x[i+1][j-1][k],x[i+1][j][k],x[i+1][j+1][k]])
                X.append(np.reshape(aux,(1, np.size(aux)))) 
        
    
        X = np.reshape(X, (np.shape(X)[0],self.input_dim*9))
        #X = X/255
        Output = self.model.predict(X, verbose = 1)
        return Output