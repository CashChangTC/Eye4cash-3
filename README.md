# Eye4cash

Folder Structure

Eye4cash---------img ( Image for readme )
           |
		   ------log ( Final model)
           |
		   ------TestDataUSD ( Data for testing)
		   |
		   ------alexnet.py ( Network model)
		   |
		   ------prediction.py ( Prediction tool)
		   |
		   ------train.py ( Train tool)
		   
# Train Steps

Launch docker avery_tensorflow

```shell
docker start avery_tensorflow
docker exec -it avery_tensorflow bash
```

This docker is using /data3 as its share folder.
If you want to train this model by yourself, 
you can using the data under /data3/Avery_Jupyter .

```shell
cd /data3/Avery_Jupyter
git clone https://github.com/AveryHu/Eye4cash.git
```

Because the train model didn't set parameters, you have to change imagefilepath (training set) and valimagefilepath (validation set) to your data location.

Then you can start training by running this py file

```shell
python train.py
```

The following img is the training lose of this final model by using the training dataset named upup7_2 under /data3/Avery_Jupyter

![image](https://github.com/AveryHu/Eye4cash/blob/master/img/trainloss.PNG?raw=true)

# Prediction Steps

Just run the prediction.py you can use this final model to do the prediction.

```
python prediction.py -p folder_to_prediction
```

Default folder to predict is TestDataUSD.

You will get the prediction results like following :

===========Prediction Results============

['1 Cent', '10 Cent', '1 Cent', '1 Cent', 'Quarter', 'Quarter', '10 Cent', 'Quarter', '10 Cent', 'Quarter', 'Quarter', '1 Cent', 'Quarter', '1 Cent', 'Qua'1 Cent', '1 Cent', 'Quarter', '10 Cent', '10 Cent', 'Quarter', 'Quarter', 'Quarter', 'Quarter']

=========================================
