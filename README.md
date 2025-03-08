# TTC Delay Prediction


In this project, we attempted to gain insight into the cause of TTC’s transit delays in 2024 using numerous machine learning techniques. In addition, our model also allows user to input the desired features to get a prediction of delay.

## Detail

- Programming Language: Python (Version 3.12)
- Package Requirements: Scikit-Learn, Numpy, Pandas, Matplotlib, Xgboost


## Files
-	Streetcar_Duration.py: the model for streetcar delay. 
-	Bus_Duration.py: the model for bus delay.
-	Streetcar_grid_search.py, bus_grid_search.py: search for the best learning rate and L2 regularization for Duration.py and Bus_Duration.py, respectively (Warning: these files could take several minutes to execute). 


## Example Usage
-	Note: Please change the data path to the correct location of the dataset files.
-	A user may want to know the potential delay of line 504 on Wednesday. The streetcar is planned to arrive at 18:00 3 minutes after the previous car and they expect a general delay of the car.  To get the prediction, at the sample user input in Streetcar_Duration.py, input:
	user_input = {
    'Date': ['03-Jan-24'],
    'Line': ['504'],
    'Time': ['18:00'],
    'Day': ['Wednesday'],
    'Incident': ['General Delay'],
    'Min Gap': '3',
    'Bound': ['']
}
which will return the delay prediction. 

-	Similarly, if a user wants to know the potential delay of bus 89 on Sunday. The bus is planned to arrive at 18:00 3 minutes after the previous car and they expect a general delay of the car.  To get the prediction, at the sample user input in Bus_Duration.py, input:
	user_input = {
    'Date': ['06-Jan-24'],
    'Route': ['89'],
    'Time': ['18:00'],
    'Day': ['Sunday'],
    'Incident': ['Security'],
    'Min Gap': '3',
}
## Video Demonstration
https://youtu.be/U6quMcwcIHg

## Team
Binhe Jia
Yukuan Zou


