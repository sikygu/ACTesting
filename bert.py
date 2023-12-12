# generate coco caption file
import json
import random
# Load the annotations file
textpath = '/home/EDA/textdata/coco2014valcaptions.txt'
outputpath = '/home/EDA/textdata/MR_coco2014valcaptions.txt'
maskfilepath = '/home/EDA/textdata/coco2014valcaptions_mask.txt'

with open('/home/t2i/cocodata/annotations_trainval2014/captions_val2014.json', 'r') as f:
    annotations = json.load(f)

# Create dictionary mapping image IDs to captions
image_to_captions = {}
for annotation in annotations['annotations']:
    image_id = annotation['image_id']
    caption = annotation['caption']
    if image_id not in image_to_captions:
        image_to_captions[image_id] = []
    image_to_captions[image_id].append(caption)

# Write captions to file
with open(textpath, 'w') as f:
    for image_id, captions in image_to_captions.items():
        caption = random.choice(captions)
        if caption.strip() == "":
            continue
        f.write(caption + '\n')


# remove empty lines
with open(textpath, 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    lines = [line[0] + line[1:].lower() for line in lines]
with open(textpath, 'w') as new_f:
    for line in lines:
        new_f.write(line + '\n')


# Open the file in read mode
with open(maskfilepath, 'r') as file:
    # Read all lines into a list
    lines = file.readlines()

# Remove the line you want to delete
lines = [line for line in lines if "[MASK]" in line]

# Open the file in write mode
with open(maskfilepath, 'w') as file:
    # Write the remaining lines to the file
    for line in lines:
        file.write(line)