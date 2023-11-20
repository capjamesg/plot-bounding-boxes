import gradio as gr
import supervision as sv
import numpy as np
from gradio.components import Image, Textbox, Radio

def plot_bounding_boxes(image, boxes, box_type):
    x0, y0, x1, y1 = [int(i.strip()) for i in boxes.split(",")]

    detections = sv.Detections(
        xyxy=np.array([[x0, y0, x1, y1]]),
        class_id=np.array([0]),
        confidence=np.array([1.0]),
    )
    # convert to cv2
    image = np.array(image)

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
    
    return annotated_image

iface = gr.Interface(
    fn=plot_bounding_boxes,
    inputs=[
        Image(type="pil", label="Image"),
        Textbox(label="Bounding Boxes", lines=3, default="100,100,200,200"),
        Radio(["xyxy", "xywh"], label="Bounding Box Type"),
    ],
    outputs=Image(type="pil"),
    title="Plot Bounding Boxes",
    description="Plot bounding boxes on an image. Useful for testing object detection models without writing bounding box code. Powered by [supervision](https://github.com/roboflow/supervision)."
)

iface.launch()