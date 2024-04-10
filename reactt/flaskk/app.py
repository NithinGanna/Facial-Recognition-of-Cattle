from flask import Flask, request, render_template, session,jsonify ,send_file
import base64
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F
import pymongo
import torchvision
from torch.utils.data import DataLoader, Dataset
import shutil
from torch.optim import Adam
import io
import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader

app = Flask(__name__)

# Define the transformation for input images (should match the one used during training)
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
    
        
# Define your model architecture
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(in_features=75 * 75 * 32, out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        output = output.view(-1, 32 * 75 * 75)
        output = self.fc(output)
        return output

folder_path = 'static/train'  # Path to your directory

if os.path.exists(folder_path) and os.path.isdir(folder_path):
    folder_count = sum(os.path.isdir(os.path.join(folder_path, name)) for name in os.listdir(folder_path))
    print(f"Number of folders in '{folder_path}': {folder_count}")
else:
    print(f"Directory '{folder_path}' does not exist or is not a directory.")

        
model = ConvNet(num_classes=folder_count)
model.load_state_dict(torch.load('models/cattle_aadhar_copy.model'))
model.eval()


@app.route('/Verify', methods=['POST'])
def upload():
    
    # handling the input image
    
    file = request.files['image']
    
    # Store the uploaded image data in the session
    # session['image_data'] = image_data
    
    # Save the uploaded file to a specific location
    # Adjust the path where you want to store the uploaded file
    # file_path = 'static/input/image.jpg'
    # file.save(file_path)  
    
    image_data = file.read()
    
        
    encoded_image = base64.b64encode(image_data).decode('utf-8')
    
    # return
    
    input_image = Image.open(file)    
    input_tensor = transform(input_image)
    input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension
    
    
    # Make predictions
    with torch.no_grad():
        output = model(input_tensor)

    # print(output)

    input_image = output.numpy()
    
    corelation_input = input_image.flatten()

    train_folder = 'static/train'  # Path to the 'train' folder
    
    b=False
    sl=[]
    cl=[]
    
    # Iterate through folders and images
    for root, dirs, files in os.walk(train_folder):
        for directory in dirs:
            folder_path = os.path.join(train_folder, directory)
            print(f"Images in folder '{directory}':")
            maxi=0
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    image_path = os.path.join(folder_path, file)
                    input_image = Image.open(image_path)
                    input_tensor = transform(input_image)
                    input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension

                    # Make predictions
                    with torch.no_grad():
                        output = model(input_tensor)
                        
                    class_checker = output.numpy()
                        
                    corelation_class = class_checker.flatten()
                        
                    similarity=np.corrcoef(corelation_input, corelation_class)[0, 1]
            
                    if((similarity.item())>0.90 and (similarity.item())>maxi):
                        b=True
                        maxi=similarity.item()
                    
                    # Perform operations with the image path (e.g., print, process, etc.)
                    # print(image_path)  # Example: Printing the image path
            
            if(b): 
                b=False
                cl.append(directory)
                sl.append(maxi)
    
    
    print(cl)   # to cheeck silimar classes
    print(sl)    # to cheeck silimarity of classes

    if(len(sl)!=0):
        b=True
    else:
        b=False

    maximum=0

    for i in sl:
        if(i>maximum):
            maximum=i
            
    # from IPython.display import Image

    if(not b):
        print("cant find this registration")
        return jsonify({"prediction":"cant find this registration"})
    else:
        # print(sl)
        print(maximum)
        ind=sl.index(maximum)
        print(ind)
        print("prediction class is ",cl[ind])
        return jsonify({"prediction":cl[ind]})
        
    
    
@app.route('/confirm',methods=['POST'])
def confirm():
    if 'image' in request.files:
        # Get the input image from the form data
        file = request.files['image']
        
        # Get the cattle ID from the form data
        cattle_id = request.form.get('cattleID')
          
        print("cattle_id",cattle_id)
        
        image_data = file.read()
        
            
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        
        # return
        
        input_image = Image.open(file)    
        input_tensor = transform(input_image)
        input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension
        
        
        # Make predictions
        with torch.no_grad():
            output = model(input_tensor)

        # print(output)

        input_image = output.numpy()
        
        corelation_input = input_image.flatten()

        train_folder = 'static/train'  # Path to the 'train' folder
        
        b=False
        sl=[]
        cl=[]
        
        specific_folder = cattle_id   # Specify the folder name you want to iterate through
        folder_path = os.path.join('static/train', specific_folder)

        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            print(f"Images in folder '{specific_folder}':")
            maxi=0
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    image_path = os.path.join(folder_path, file)
                    # Perform operations with the image path (e.g., print, process, etc.)
                    print(image_path)  # Example: Printing the image path
            
                    input_image = Image.open(image_path)
                    input_tensor = transform(input_image)
                    input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimensio
                    # Make predictions
                    with torch.no_grad():
                        output = model(input_tensor)
                        
                    class_checker = output.numpy()
                        
                    corelation_class = class_checker.flatten()
                        
                    similarity=np.corrcoef(corelation_input, corelation_class)[0, 1]
            
                    if((similarity.item())>0.90 and (similarity.item())>maxi):
                        b=True
                        maxi=similarity.item()
                    
                    # Perform operations with the image path (e.g., print, process, etc.)
                    # print(image_path)  # Example: Printing the image path
            
            if(b): 
                b=False
                cl.append(cattle_id)
                sl.append(maxi)
                
            print(cl)   # to cheeck silimar classes
            print(sl)    # to cheeck silimarity of classes

            if(len(sl)!=0):
                b=True
            else:
                b=False

            maximum=0

            for i in sl:
                if(i>maximum):
                    maximum=i
                    
            # from IPython.display import Image

            if(not b):
                print("cant find this registration")
                return jsonify({"prediction":"Not Matched"})
            else:
                # print(sl)
                print(maximum)
                ind=sl.index(maximum)
                print(ind)
                print("prediction class is ",cl[ind])
                if(cl[ind]==cattle_id):
                    return jsonify({"prediction":cl[ind]})
                else:
                    return jsonify({"prediction":"Not Matched"})
            
        else:
            return jsonify({"prediction":"id does not exist"})
            
    else:
        return jsonify({"prediction":'No image found in the request'})
    

# Function to get the ID of the last document in the collection
def get_last_cattle_id():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["Display"]
    collection = db["cows"]
    last_document = collection.find_one(sort=[('cattle_id', pymongo.DESCENDING)])
    if last_document:
        return int(last_document['cattle_id'])
    print("1 for no last document")
    return 1  # Return 1 if no documents are found

@app.route('/upload_images', methods=['POST'])
def upload_images():
    folder_name = str(get_last_cattle_id())
    upload_path = os.path.join( 'static', 'new_added', folder_name)
    
    if not os.path.exists(upload_path):
        os.makedirs(upload_path)

    # Save uploaded images in the folder
    for key, file in request.files.items():
        if file.filename != '':
            file.save(os.path.join(upload_path, file.filename))

    print('Images uploaded successfully')
    
    # Define your transformation
    transformer = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Path for the main dataset directory
    dataset_path = "static"

    # Define a custom dataset class
    class CustomDataset(Dataset):
        def __init__(self, data_dir, transform=None):
            self.data = torchvision.datasets.ImageFolder(data_dir, transform=transform)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # Create the custom datasets
    train_dataset = CustomDataset(os.path.join(dataset_path,'train'), transform=transformer)

    # Create data loaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # for adding new registration
    
    #checking for device
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Define your model architecture
    class ConvNet(nn.Module):
        def __init__(self, num_classes):
            super(ConvNet, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(num_features=12)
            self.relu1 = nn.ReLU()
            self.pool = nn.MaxPool2d(kernel_size=2)
            self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()
            self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(num_features=32)
            self.relu3 = nn.ReLU()
            self.fc = nn.Linear(in_features=75 * 75 * 32, out_features=num_classes)

        def forward(self, input):
            output = self.conv1(input)
            output = self.bn1(output)
            output = self.relu1(output)
            output = self.pool(output)
            output = self.conv2(output)
            output = self.relu2(output)
            output = self.conv3(output)
            output = self.bn3(output)
            output = self.relu3(output)
            output = output.view(-1, 32 * 75 * 75)
            output = self.fc(output)
            return output
    
    # Path to the existing model
    model_path = "models/cattle_aadhar_copy.model"

    # Path for the new data
    new_data_path = "static/new_added"
    
    print("after model path",folder_count)
    
    try:
        # Load the existing model
        model = ConvNet(num_classes=folder_count).to(device)
        
        print("after model")
        
        model.load_state_dict(torch.load(model_path))
        
        print("afer loading model")
        
    except Exception as e:
        print("Error of loading and creating model:", e)
    
    print("before combined num classes")

    # Update the model's architecture for the combined dataset (146 old IDs + 1 new ID)
    combined_num_classes = folder_count +1  # 146 old cattle IDs + 1 new cattle ID (250)
    model.fc = nn.Linear(in_features=75 * 75 * 32, out_features=combined_num_classes)
    
    print("before loss function")

    # Define the optimizer and loss function
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()
    
    print("after loss function")

    # Combine the old and new datasets using a custom dataset class
    class CombinedDataset(Dataset):
        def __init__(self, dataset1, dataset2):
            self.dataset1 = dataset1
            self.dataset2 = dataset2

        def __len__(self):
            return len(self.dataset1) + len(self.dataset2)

        def __getitem__(self, idx):
            if idx < len(self.dataset1):
                return self.dataset1[idx]
            else:
                return self.dataset2[idx - len(self.dataset1)]
            
    new_data = CustomDataset(os.path.join(dataset_path, 'new_added'), transform=transformer)

    # Create the combined dataset
    combined_dataset = CombinedDataset(train_dataset, new_data)
    combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)
    
    print("before training")

    # Fine-tune the model on the combined dataset
    model.train()
    
    print("before epochs")

    for epoch in range(10):  # Adjust the number of fine-tuning epochs
        train_accuracy = 0.0
        train_loss = 0.0

        for i, (images, labels) in enumerate(combined_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, prediction = torch.max(outputs.data, 1)
            train_accuracy += int(torch.sum(prediction == labels.data))

        train_accuracy = train_accuracy / len(combined_dataset)
        train_loss = train_loss / len(combined_dataset)

        print(f'Fine-Tuning Epoch: {epoch}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}')

    # Save the updated model
    torch.save(model.state_dict(), 'cattle_aadhar_updated.model')
    
    # Delete the old model
    old_model_path = 'models/cattle_aadhar_copy.model'
    if os.path.exists(old_model_path):
        os.remove(old_model_path)
        print(f"Deleted old model: {old_model_path}")

    # Rename the updated model
    updated_model_path = 'cattle_aadhar_updated.model'
    new_model_path = 'models/cattle_aadhar_copy.model'
    os.rename(updated_model_path, new_model_path)
    print(f"Renamed updated model to: {new_model_path}")
    
    # to check what folders are present in new_added folder

    directory_path = "static/new_added"  # Replace with the path to your directory

    # Get a list of all items (files and folders) in the directory
    items = os.listdir(directory_path)

    # Filter the items to get only the folders (directories)
    folders = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]

    # Print the list of folders
    for folder in folders:
        f=folder
    print(f)
    
    # to move the newly trained cattle to already trained cattles

    source_path = "static/new_added"

    source_folder = os.path.join(source_path,f)  # Replace with the path to the source folder
    destination_folder = "static/train"  # Replace with the path to the destination folder

    # Move the '250_muzzle' folder to the destination folder
    shutil.move(source_folder, destination_folder)

    return jsonify({"progress": get_last_cattle_id()})
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["Display"]
    collection = db["cows"]

    # Find the document with the highest cattle_id
    last_document = collection.find_one(sort=[('cattle_id', pymongo.DESCENDING)])

    if last_document:
        return last_document['cattle_id']
    return 0  # Return 0 if no documents are found

@app.route('/submit_form', methods=['POST'])
def submit_form():
    try:
        form_data = request.json  # Assuming form data is sent as JSON

        new_cattle_id = int(get_last_cattle_id())+1
                                                                                                        
        # Add the new form data to MongoDB with the incremented ID
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["Display"]
        collection = db["cows"]
        
        data=form_data['image']
        
        st = "data:image/jpeg;base64,"

        # Find the position of the comma after 'base64,'
        # index = data.find('base64,') + len(data)

        # Extract the front part containing image information
        form_data['image'] = data[len(st):]

        form_data['cattle_id'] = str(new_cattle_id)  # Add the new ID to form data
        collection.insert_one(form_data)
    
        print('Form submitted successfully of cattle_id', new_cattle_id)    

        return jsonify({'message': 'Form submitted successfully!', 'cattle_id': new_cattle_id})
    except Exception as e:
        print('Form not submitted of cattle_id', new_cattle_id)
        return jsonify({'message': 'Error in submitting form data'}), 500


class YOLO_Pred():
    def __init__(self,onnx_model,data_yaml):
        # load YAML
        with open(data_yaml,mode='r') as f:
            data_yaml = yaml.load(f,Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']
        
        # load YOLO model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                 
    def predictions(self,image):
        
        row, col, d = image.shape
        # get the YOLO prediction from the the image
        # step-1 convert image into square image (array)
        max_rc = max(row,col)
        input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
        input_image[0:row,0:col] = image
        # step-2: get prediction from square array
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WH_YOLO,INPUT_WH_YOLO),swapRB=True,crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward() # detection or prediction from YOLO

        # Non Maximum Supression
        # step-1: filter detection based on confidence (0.4) and probability score (0.25)
        detections = preds[0]
        boxes = []
        confidences = []
        classes = []

        # widht and height of the image (input_image)
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w/INPUT_WH_YOLO
        y_factor = image_h/INPUT_WH_YOLO

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4] # confidence of detection an object
            if confidence > 0.4:
                class_score = row[5:].max() # maximum probability from 20 objects
                class_id = row[5:].argmax() # get the index position at which max probabilty occur

                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    # construct bounding from four values
                    # left, top, width and height
                    left = int((cx - 0.5*w)*x_factor)
                    top = int((cy - 0.5*h)*y_factor)
                    width = int(w*x_factor)
                    height = int(h*y_factor)

                    box = np.array([left,top,width,height])

                    # append values into the list
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        # clean
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        # NMS
    #         index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45).flatten()
        index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)
        index = np.array(index).flatten()

        muzzle_detected = False
        
        # Draw the Bounding
        for ind in index:
            # extract bounding box
            x,y,w,h = boxes_np[ind]
            bb_conf = int(confidences_np[ind]*100)
            classes_id = classes[ind]
            class_name = self.labels[classes_id]
            colors = self.generate_colors(classes_id)

            text = f'{class_name}: {bb_conf}%'

            cv2.rectangle(image,(x,y),(x+w,y+h),colors,2)
            cv2.rectangle(image,(x,y-30),(x+w,y),colors,-1)

            cv2.putText(image,text,(x,y-10),cv2.FONT_HERSHEY_PLAIN,0.7,(0,0,0),1)
            
            if class_name== 'muzzles' or 'muzzle':
                muzzle_detected = True


        if muzzle_detected:
            return image[y:y+h, x:x+w]  # for extracting the only muzzle part  
        else:
            return None


    def generate_colors(self,ID):
        np.random.seed(10)
        colors = np.random.randint(100,255,size=(self.nc,3)).tolist()
        return tuple(colors[ID])


@app.route('/yolo_crop', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return "No image uploaded", 400

    image_file = request.files['image']
    if image_file.filename == '':
        return "No selected image", 400

    # Save the uploaded image temporarily
    in_image = Image.open(image_file)
    # Perform processing on the image (e.g., image manipulation)
        
    yolo = YOLO_Pred("yolo//muzzle_yolo_predictions//Model10-20231020T053132Z-001//Model10//weights//best.onnx",
                     "yolo//data.yaml")
    
    def fun(image):
        
        if image is None or image.size == 0:
            print("Invalid or empty image. Cannot process.")
        else:       
            # Check if the passed image is valid
            if image is None or image.size == 0:
                print("Invalid or empty image. Cannot process.")
                return None  # Return None indicating failure
            
            else:
                if image is None or image.size == 0:
                    print("Invalid or empty image. Cannot process.")
                else:
                    # resized_img = cv2.resize(image,(600, 600))  # Convert PIL Image to NumPy array
                    resized_img = cv2.resize(np.array(image), (600, 600))
                    processed_imge = yolo.predictions(resized_img)

                    if processed_imge is None:
                        print("image is None")
                        return None
                    elif processed_imge.size==0:
                        print("image.size==0")
                        return None
                    elif processed_imge is not None:
                        print("got cropped")
                        return processed_imge
        
    processed_img=fun(in_image)
    
    print("image")
    
    if processed_img is None:
        print("processsed img is None")
        return "Failed to process image", 500
    
    print("yolo crop done")
    
    # Read the image file and encode it as base64
    # image_data = processed_img.read()
    # encoded_image = base64.b64encode(image_data).decode('utf-8')
    _, buffer = cv2.imencode('.png', processed_img)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    # Return the base64 string as a JSON response
    return jsonify({'base64Data': encoded_image})

