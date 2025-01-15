# Model-fusion-gesture-recognition-based-on-LSTM-and-Mediapipe

This project extracts feature points based on the Mediapipe open-source model, uses LSTM, CNN, and Vision of Transformer network models for training to obtain a deep learning model, and finally deploys and applies it.

## 1.Environment

| Package Name     | Version   | Package Name     | Version   |
|------------------|-----------|------------------|-----------|
| opencv-python    | 4.9.0.80  | timm             | 0.9.2     |
| pillow           | 10.2.0    | torchvision      | 0.19.1    |
| torch            | 2.4.1     | Python           | 3.11.9    |
| mediapipe        | 0.10.14   | CUDA             | 12.7      |
| einops           | 0.8.0     | NVIDIA-SMI       | 566.14    |
| h5py             | 3.11.0    | scikit-learn     | 1.4.2     |

## 2.Usage

Our code uses the feature point extraction technology based on mediapipe, and uses the relevant code in utils.py to package the generated data set. Then we use CNN + LSTM and ViT + LSTM technologies to train and test the data set respectively. Finally, the obtained model files are deployed for real-time recognition of dynamic gestures.
Among them, our project designed two GUI interfaces for data collection and gesture recognition to facilitate the processing of related data, namely APP_Data_Collector.py and APP_Gesture_Recognizer.py. Of course, you can also run both interfaces at the same time by running APP_Redal.py for operation. If you try to input a new gesture, you can collect about 50 gesture videos of yours lasting for 1.5 seconds through the data collection interface, then extract the key point data through utils.py, and finally obtain a new network model through train.py.

### Data Collection

![Data Collection](asserts\数据采集.png)  
First of all, you can click "New Class Name" to add new gesture categories. Then the default collection time of the system is 1.5 seconds. You need to complete the gesture collection within 1.5 seconds and repeat the collection of the same gesture 50 times. When it's done, run utils.py to complete the collection of the key point coordinates of the video gesture, and the system will automatically save the data under the embed/ folder.

### Gesture Recognition

Finally, it's about the deployment of the trained models. This model used CNN, PointNet, Transformer, and LSTM and obtained three models. Finally, the model results were fused and output to obtain the final gesture classification result. Of course, the relevant gesture classification demonstrations I have uploaded to BiliBili, and you can watch it directly through the link. Finally, there are two choices of gesture recognition models, and you can check this by running the program.
    #### python APP_Redal.py
![Running GUI](asserts\静态手势.png)
![Running GUI](asserts\动态手势.png)

## 3.Model

The three models implemented in this project all display the relevant network structure in the `.onnx` format. You can also generate the `.onnx` file by running the relevant code under `models/` and visualize the model results through the [Netron app](https://netron.app).

|![CNN_LSTM](asserts\CNN_LSTM.onnx.png) | ![PointNet_LSTM](asserts\PointNet_LSTM.onnx.png) | ![Transformer_LSTM](asserts\Transformer_LSTM.onnx.png) |
|---------------------------------------|--------------------------------------------------|--------------------------------------------------------|

## 4.Training & Testing

Next, you can train the network model by running the train.py code file. And the relevant test code is also in the same folder. You can use the plot_confusion_matrix function in the code to draw the confusion matrix and classification score of the test.The following is the display of the confusion matrices of the three model files of this project that I present for your reference and model selection.  
![Confusion Matrix](asserts\CNN_LSTM_CM.png)  
![Confusion Matrix](asserts\PointNet_LSTM_CM.png)  
![Confusion Matrix](asserts\Transformer_LSTM_CM.png)  

## 5.Thanks

Finally, Thank you for watching. Please could you light up a little star.
