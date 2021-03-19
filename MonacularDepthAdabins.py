from AdaBins.infer import InferenceHelper
from PIL import Image

def get_depth(imagepath):
    infer_helper = InferenceHelper(dataset='kitti')
    pil_img = Image.open(imagepath)
    pil_img = pil_img.resize((640,480))
    bin_centers, predicted_depth, depth_viz = infer_helper.predict_pil(pil_img, visualized=True)
    return bin_centers, predicted_depth, depth_viz
