from fastai.vision import *
path = Path('models')
img = open_image(path/'image.jpg')
learn = load_learner(path)
pred_class,pred_idx,outputs = learn.predict(img)
print(str(pred_class))