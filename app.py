import imghdr
import os
import shutil
import cv2
from keras.src.utils.image_utils import img_to_array
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
#import ModelBuilder as I
import zipfile
import random
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Rescaling
from keras.utils import image_dataset_from_directory
import re
from pathlib import Path


# Initialize session state for modal status
if 'modal_open' not in st.session_state:
    st.session_state.modal_open = False

# Model runs
runs = 10

@st.dialog("Select an Image:")
def open_dialog():
    # Sample image URLs or paths
    images = []
    data_dir = 'test_images'
    for image in os.listdir(data_dir):
        images.append(image)

    # Titles for buttons (optional)
    titles = [f"Image {i + 1}" for i in range(len(images))]

    # Grid dimensions
    cols_per_row = 3

    st.title("Test Images Grid")

    # Loop through the images and buttons
    for i in range(0, len(images), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if i + j < len(images):
                with cols[j]:
                    # Resize the image to a specific dimension
                    new_size = (600, 600)  # (width, height)
                    testImage = Image.open(data_dir + "/" + images[i + j])
                    resized_image = testImage.resize(new_size)

                    st.image(resized_image, use_container_width=True)
                    if st.button(titles[i + j], key=f"btn_{i + j}"):
                        st.session_state.image = data_dir + "/" + images[i + j]
                        st.session_state.modal_open = False
                        st.rerun()  # This closes the dialog by rerunning the app

# Inject custom CSS
st.markdown(
    """
    <style>
    .stApp {
        /*background-color: #353434 ;*/
        background-color: #ffffff;

    }
    </style>
    """,
    unsafe_allow_html=True
)
# Replace with your image URL or local path
image_url = "https://deepvisionai.in/img/fac.jpg"

# HTML/CSS to center and style the image
st.markdown(
    f"""
    <style>
    .image-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }}
    .responsive-img {{
        height: 200px;
        width: 400px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}
    </style>
    <div class="image-container">
        <img class="responsive-img" src="{image_url}" alt="Styled Image">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #ad84eb;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border-color: white
    }
    </style>
    """, unsafe_allow_html=True)

# Create buttons with st.button

# Streamlit UI
st.title(f":gray[SnapAI: Smart Image Detection]")
#st.subheader("Add Your Name Here")
st.write(f"### :gray[Upload an image and let the model predict the class.]")

# File uploader:blue
uploaded_file = st.file_uploader(f"### :gray[Upload a zip...]", type=[".zip"])
class_names = []

# Create Buttons
col1, col2, col3, col4 = st.columns([1,1,1,1])

with col1:
    button1 = st.button('Build Model')
with col2:
    button2 = st.button('Test Images')
with col3:
    button3 = st.button('New Images')
with col4:
    button4 = st.button('Delete Data')

# Get class names from your training data (same order as before)
if button1:
    testImageDir = 'test_images/'
    dataDir = 'data'
    if os.path.isdir(testImageDir) and os.path.isdir(dataDir):
        with st.spinner("### Processing..."):
            progress = st.progress(0)

        st.info(f"### :gray[Building Your Model.....Please Wait]")
        # Path to your dataset directory
        for i in range(5):
            progress.progress(i + 1)
        data_dir = "data"

        image_exts = ['jpeg', 'jpg', 'bmp', 'png']

        print("Here are your categories: ")
        print(os.listdir(data_dir))


        for image_class in os.listdir(data_dir):
            for image in os.listdir(os.path.join(data_dir, image_class)):
                image_path = os.path.join(data_dir, image_class, image)
                try:
                    img = cv2.imread(image_path)
                    with Image.open(image_path) as img:
                        img.load()              # Fully decode the image
                        tip = img.format.lower()  # "jpeg", "png", "bmp", etc.
                    if tip not in image_exts:
                        print('Image not in ext list {}'.format(image_path))
                        os.remove(image_path)
                except Exception as e:
                    print('Issue with image {}'.format(image_path))
                    # os.remove(image_path)
        for i in range(10):
            progress.progress(i + 5)

        # Parameters
        batch_size = 32
        img_height = 224
        img_width = 224
        seed = 42

        # Load dataset from folders
        train_ds = image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=seed,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

        val_ds = image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=seed,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

        # Get class names
        class_names = train_ds.class_names
        print("Class names:", class_names)
        for i in range(15):
            progress.progress(i + 15)

        # Prefetching (performance improvement)
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

        # Build the model (simple CNN)
        model = Sequential([
            Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
            Conv2D(32, 3, activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, activation='relu'),
            MaxPooling2D(),
            Conv2D(128, 3, activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(len(class_names), activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        for i in range(20):
            progress.progress(i + 30)

        # Train the model
        epochs = runs
        for i in range(25):
            progress.progress(i + 50)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )

        model.save("model.h5")

        for i in range(25):
            progress.progress(i + 76)
        st.success("All Done!")

if button2:
    testImageDir = 'test_images/'
    dataDir = 'data'
    if os.path.isdir(testImageDir) and os.path.isdir(dataDir):
        st.session_state.modal_open = True
        open_dialog()
        # Track if the modal was previously open
        if st.session_state.modal_open:
            st.session_state.modal_was_open = True

if not st.session_state.modal_open and st.session_state.get('modal_was_open', False):
    # Display uploaded image
    image = Image.open(st.session_state.image)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    class_names = os.listdir("data")
    st.write(class_names)

    # Preprocess image
    img_height, img_width = 224, 224
    image = image.resize((img_width, img_height))
    img_array = img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    # Load the trained model
    model = tf.keras.models.load_model("model.h5")

    # Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions)
    st.write(predictions)
    st.write(score)
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.write(f"### :gray[**Prediction:**] :gray[{predicted_class}]")
    st.write(f"### :gray[**Confidence:**] :gray[{confidence:.2f}%]")

if button3:
    imageCounts = []
    testImageDir = 'test_images/'
    dataDir = 'data'
    zipDir = 'zipFiles'
    if os.path.isdir(testImageDir) and os.path.isdir(dataDir):
        for dir in os.listdir(dataDir):
            imageCount = 0
            dirPath = dataDir + '/' + str(dir)
            for img in os.listdir(dirPath):
                imgPath = dirPath + '/' + str(img)
                os.remove(imgPath)
                imageCount += 1
            os.rmdir(dirPath)
            imageCounts.append(imageCount)
        for img in os.listdir(testImageDir):
            imgPath = testImageDir + '/' + str(img)
            os.remove(imgPath)
        for zipFile in os.listdir(zipDir):
            extract_to = os.path.join(dataDir, str(os.path.splitext(zipFile)[0]))
            zipPath = os.path.join(zipDir, zipFile)
            # Unzip the file
            with zipfile.ZipFile(zipPath, 'r') as zip_ref:
                zip_ref.extractall(extract_to)

        image_exts = ['jpeg', 'jpg', 'bmp', 'png']
        classCount = 0
        for image_class in os.listdir(dataDir):
            imageClassNumber = []
            for x in range(10):
                imageClassNumber.append(random.randint(0,imageCounts[classCount]))
            count = -1
            imagesLoaded = 0
            allImagesLoaded = False
            for image in os.listdir(os.path.join(dataDir, image_class)):
                count += 1
                for num in imageClassNumber:
                    if count == num:
                        image_path = os.path.join(dataDir, image_class, image)
                        try:
                            img = cv2.imread(image_path)
                            with Image.open(image_path) as img:
                                img.load()              # Fully decode the image
                                tip = img.format.lower()  # "jpeg", "png", "bmp", etc.
                            if tip in image_exts:
                                dst_folder = "test_images"
                                os.makedirs(testImageDir, exist_ok=True)
                                #print("Image Path: " + str(image_path))
                                shutil.move(image_path, testImageDir)
                                imagesLoaded += 1
                                if imagesLoaded == 3:
                                    allImagesLoaded = True
                                    break
                        except Exception as e:
                            message = 'Issue with image {}'.format(image_path)
                if allImagesLoaded:
                    break
        st.success("Images Reset")
        
if button4:
    testImageDir = 'test_images/'
    dataDir = 'data'
    zipDir = 'zipFiles'
    if os.path.exists("model.h5"):
        os.remove("model.h5")
        st.success('Model Deleted')
    if os.path.isdir(dataDir):
        for dir in os.listdir(dataDir):
            dirPath = dataDir + '/' + str(dir)
            for img in os.listdir(dirPath):
                imgPath = dirPath + '/' + str(img)
                os.remove(imgPath)
            os.rmdir(dirPath)
        os.rmdir(dataDir)
        st.success('Data Deleted')
    if os.path.isdir(testImageDir):
        for img in os.listdir(testImageDir):
            imgPath = testImageDir + '/' + str(img)
            os.remove(imgPath)
        os.rmdir(testImageDir)
        st.success('Test Data Deleted')
    if os.path.isdir(zipDir):
        for zipFile in os.listdir(zipDir):
            zipPath = zipDir + '/' + str(zipFile)
            os.remove(zipPath)
        os.rmdir(zipDir)
        st.success('ZIP Files Deleted')
    st.success('Data Delete Complete')

data_dir = 'data/'
zip_dir = 'zipFiles/'
if uploaded_file is not None:
    image_exts = ['jpeg', 'jpg', 'bmp', 'png']
    if uploaded_file.name.lower().endswith(".zip") and not(os.path.isdir(data_dir + str(os.path.splitext(uploaded_file.name)[0]))):
        extract_to = data_dir + str(os.path.splitext(uploaded_file.name)[0])

        # Create the directories if it doesn't exist
        os.makedirs(extract_to, exist_ok=True)
        os.makedirs(zip_dir, exist_ok=True)

        save_path = os.path.join(zip_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        # Unzip the file
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        MIN_SIZE = 1024  # 1 KB

        for file in Path(data_dir).rglob("*"):
            if file.is_file() and file.suffix.lower() in image_exts:
                if file.stat().st_size < MIN_SIZE:
                    print(f"Deleting tiny file: {file}")
                    os.remove(file)

        image_exts = ['jpeg', 'jpg', 'bmp', 'png']
        for image_class in os.listdir(data_dir):
            if image_class == str(os.path.splitext(uploaded_file.name)[0]):
                imageNotFound = True
                isTestImage = True
                testImageCount = 0
                count = 0
                for image in os.listdir(os.path.join(data_dir, image_class)):
                    SAFE_FILENAME = re.compile(r'^[A-Za-z0-9 _().-]+$')

                    if not SAFE_FILENAME.match(image):
                        print(f"Skipping invalid filename: {image}")
                        continue
                    image_path = os.path.join(data_dir, image_class, image)
                        
                    # Verify this is a real image
                    try:
                        with Image.open(image_path) as img:
                            img.load()   # Raises an exception if the file is not a valid image
                    except Exception:
                        print(f"Skipping invalid image: {image_path}")
                        os.remove(image_path)          # or continue if you don't want to delete it
                        continue
                    for x in range(2):
                        image_path = os.path.join(data_dir, image_class, image)
                        
                        img = os.path.splitext(image)[0]
                        ext = os.path.splitext(image)[1]
                        newImage = str(img) + str(x) + str(ext)
                        newImagePath = data_dir + image_class + "/" +  newImage
                        oldImagePath = data_dir + image_class + "/" +  image                        
                        testImageCount += 1
                        
                        if testImageCount > 6:
                            isTestImage = False
                        
                        if not(isTestImage) and os.path.exists(oldImagePath):
                            shutil.copy(oldImagePath, newImagePath)
                        
                        try:
                            img = cv2.imread(image_path)
                            with Image.open(image_path) as img:
                                img.load()              # Fully decode the image
                                tip = img.format.lower()  # "jpeg", "png", "bmp", etc.
                            if tip in image_exts and imageNotFound:
                                dst_folder = "test_images/"
                                os.makedirs(dst_folder, exist_ok=True)
                                #print("Image Path: " + str(image_path))
                                shutil.move(image_path,dst_folder)
                                count += 1
                                if(count == 3):
                                    imageNotFound = False
                                    st.session_state.uploader_key = 0
                                    break
                        except Exception as e:
                            message = 'Issue with image {}'.format(image_path)
                            
                            
    st.success('Upload Complete!')
                            
            
