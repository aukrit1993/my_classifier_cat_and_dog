import numpy as np
from keras.preprocessing import image
from keras.models import load_model
classifier = load_model('test_model.h5')
test_image = image.load_img('dataset/single_prediction/cat_test_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(result[0][0])
prediction = ''
if result[0][0] == 1:
    prediction = 'This is a dog'
    
else:
    prediction = 'This is a cat'
    
print(prediction)