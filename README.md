# Autism Detection System
A web-based application that leverages deep learning and behavioral analysis to assess the risk of Autism Spectrum Disorder (ASD) in children. The system combines image-based facial analysis with a clinically inspired quiz to provide a comprehensive screening report.

_**Features**_

**Image Analysis**: Uses a fine-tuned ResNet50 CNN model to classify uploaded facial images as High or Low autism risk. 

**Autism Quiz**: A behavioral screening quiz based on clinical observations.

**Combined Assessment**: Integrates quiz and image results to produce a final risk level.

**PDF Report Generation**: Generates downloadable reports for record-keeping or clinical follow-up.

**User-Friendly Interface**: Built using Flask with an intuitive UI for easy navigation.


_**Tech Stack**_

**Frontend**: HTML, CSS, JavaScript (Flask templates)

**Backend**: Python, Flask

**Model**: TensorFlow/Keras with ResNet50

**Other**: OpenCV, ReportLab, NumPy

_**Model Download**_

To use the image detection feature, download the pre-trained .h5 model:

📥 https://drive.google.com/file/d/1s5OimbbO_ZRaRgUSTyETKdBeWK4tPeVU/view?usp=drive_link

