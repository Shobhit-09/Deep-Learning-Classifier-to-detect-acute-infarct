This readme will guide you through the execution steps of the entire project by group 8 for Practical approach to Deep Learning.

The initial dataset was provided by the professor as the part of assignment 2.
In assignment 2, we cleaned the data by seperating the images into disease specific folders.

This was saved in the CLEANED_DATA folder.

FINAL PROJECT:

(Part A)

To execute part A of code, we need Jupyter notebook.
	
	Step 1:	Open Jupyter notebook and install required libraries like shutil, matplotlib, cv2.
	Step 2:	Run the script try.ipynb on Jupyter notebook.
	Step 3:	Create the dictionary and list of disease names in first cell for testing/training the model.
	Step 4: In Second cell,we split the dataset into train and test data.
		train data -> "TRAIN_DATA" folder
		test data  -> "TEST_DATA" folder
	Step 5: Now import the necessary libraries in third cell that are required.
	Step 6:	Then we create a list for training and testing which contains grayscale images with their respective disease names.
	Step 7:	In the project directory, the augmented data is then stored in "aug_data" folder which results in increasing the amount of data and makes the model more robust to slight 		variations
	Step 8:	In next cell, we describe our CNN structure with multiple layers.
	Step 9:	We then one-hot encode our labels as strings can't be fed into the network.
	Step 10:In next cell, we train our model on augmented data created in Step 7.
	Step 11:In the next 2 cells, we predict the accuracy of our model on train data as well as the test data.
	Step 12:The model is then saved as a h5 file and the same is used to obtain our pb file.
	Step 13:The pb file is then given to our model optimizer to generate the xml and bin files.
	 
	
(Part B)

	Step 1:	Open the command prompt and open bin location
		>> cd C:\Program Files (x86)\IntelSWTools\openvino\bin
	Step 2:	Initialise the openvino environment by running the following command:
		>> setupvars.bat
	Step 3:	Go back to the project directory by using the command:
		>> cd C:\Users\Dell\Desktop\DL
	Step 4:	Finally we need to run the inference script final.py by:
		>> python final.py -i "<location of input image>" -x "<location of xml file>" -b "<location of bin file>" --labels "<location of file that stores the labels>"
	Step 5: Run the script "inference_accuracy.py" to perform inference on the entire inference set and get the accuracy by typing in.
		>> python inference_accuracy.py
END	
