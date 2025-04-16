from keras.layers import TFSMLayer

model = TFSMLayer("wildlenswebui/model/v2/", call_endpoint="serving_default")