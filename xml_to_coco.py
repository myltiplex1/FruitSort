import os
import xml.etree.ElementTree as ET

# === CONFIGURATION ===
classes = ["apple", "banana", "orange"]  # Your object categories

# Set your input/output directories
xml_dir = r"C:\Users\user\Downloads\yolovfruit\yolov5\dataset\annots\val"    # XML files
output_label_dir = r"C:\Users\user\Downloads\yolovfruit\yolov5\dataset\labels\val"  # YOLO .txt labels

# Create the label output folder if it doesn't exist
os.makedirs(output_label_dir, exist_ok=True)

# === Convert VOC (xmin, ymin, xmax, ymax) to YOLO (x_center, y_center, width, height), normalized ===
def convert_box(size, box):
    dw = 1.0 / size[0]  # 1 / image width
    dh = 1.0 / size[1]  # 1 / image height
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    width = box[2] - box[0]
    height = box[3] - box[1]
    return (x_center * dw, y_center * dh, width * dw, height * dh)

# === Parse a single XML file and write the corresponding YOLO .txt file ===
def convert_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    image_width = int(size.find("width").text)
    image_height = int(size.find("height").text)

    # Use the base name of the XML file (e.g., 001.xml → 001.txt)
    base_name = os.path.splitext(os.path.basename(xml_path))[0]
    txt_path = os.path.join(output_label_dir, base_name + ".txt")

    with open(txt_path, "w") as out_file:
        for obj in root.findall("object"):
            cls = obj.find("name").text.strip().lower()
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find("bndbox")
            bbox = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymax").text),
            )
            yolo_bbox = convert_box((image_width, image_height), bbox)
            out_file.write(f"{cls_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")

# === Process all XML files in the folder ===
def convert_all_annotations(xml_folder):
    count = 0
    for file in os.listdir(xml_folder):
        if file.endswith(".xml"):
            full_path = os.path.join(xml_folder, file)
            convert_annotation(full_path)
            count += 1
    print(f"✅ Converted {count} XML files to YOLO format.")

# === Run conversion ===
if __name__ == "__main__":
    convert_all_annotations(xml_dir)
