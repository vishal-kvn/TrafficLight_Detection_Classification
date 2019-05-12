# TrafficLight_Detection_Classification

Transfer learning has been used to detect and classify traffic lights. The following models were cosndiderd -
* (ssd_mobilenet_v1_coco_2018_01_28)[http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz]
* (faster_rcnn_resnet101_coco_2018_01_28)[http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz]

## Model Training Steps

An [AWS Deep Learning AMI](https://aws.amazon.com/marketplace/pp/B077GCH38C) has been used for training the model. The following steps were used -

### Installing Dependencies
The training dependencies can be installed by following the steps outlined on (this page)[https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md]

### Download [Tensorflow models](https://github.com/tensorflow/models)
```
git clone https://github.com/tensorflow/models.git
cd models/research

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

cd object_detection
```

### Training data
There are 2 options for training data. Download [Training data](https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/view) or
build training data by using annotation tools like [LabelImg](https://github.com/tzutalin/labelImg) and include the TFrecord file in the
`./data` folder

### Download models from Tensor flow training zoo to start transfer learning 
```
# Ensure that the models are downloaded into the /research/object_detection folder
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz

unzip ssd_mobilenet_v1_coco_2018_01_28.tar.gz
```

### Download required files
```
# Verify that the current directory is (/models/research/object_detection)
wget https://raw.githubusercontent.com/vishal-kvn/TrafficLight_Detection_Classification/master/config/ssd_mobilenet_v1.config

cd /data

wget https://raw.githubusercontent.com/vishal-kvn/TrafficLight_Detection_Classification/master/data/traffic_light_label_map.pbtxt
```

### Run training
```
python model_main.py \
--pipeline_config_path='./ssd_mobilenet_v1.config' \
--model_dir='training_ssd_mobilenet_v1_sim/' \
--sample_1_of_n_eval_examples=1 \
--alsologtostderr

```

### Export inference graph
```
python export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path ./ssd_mobilenet_v1.config \
--trained_checkpoint_prefix training_ssd_mobilenet_v1_sim/model.ckpt-5000\
--output_directory ssd_mobilenet_v1_inference_graph
```

### Validate model using the notebook
```
jupyter notebook --ip=0.0.0.0 --no-browser #Ensure that a rule is added to the security group to accept inbound traffic on port 8888.
```
Copy the IP address of the ec2 instance and view the notebook and run all cells
