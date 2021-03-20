# cs231aApproachingOdt
This is the final code submission to CS231A project, Approaching Object Detection. 
This requires to be have Yolo, and Adabins checked out in the same folder structure. 

Thank you for the great work from these repositories.
- YoloV3: https://github.com/eriklindernoren/PyTorch-YOLOv3.git
- Adabins: https://github.com/shariqfarooq123/AdaBins

Detecting approaching objects is a crucial in many scenarios and has wide applications from detecting self driving cars to warning blind people of any approaching object. In this project we make a unified prediction using 2D bounding box detections, optical flows and monocular depth estimation to make a prediction if the 2D bounding box is approaching or leaving. Tracking is done using template base matching in a 'tracking-by-detection' paradigm. This method is evaluated on separately annotated ground truth for the approaching objects.

Sample output are in the output folder.

