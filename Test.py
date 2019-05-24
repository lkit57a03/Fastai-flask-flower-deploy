from fastai.vision import *
path = Path('models')
img = open_image(path/'102841525_bd6628ae3c.jpg')
learn = load_learner(path)
pred_class,pred_idx,outputs = learn.predict(img)
print(pred_class)