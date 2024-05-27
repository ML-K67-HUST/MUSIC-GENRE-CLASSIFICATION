# Noi dung report o day
## Acknowledgement <br>
// Tuan Anh Viet

## Abstract <br>
//Khang viet
<p>
In this report, we present a comprehensive approach to developing a machine learning model capable of classifying songs into predefined genres. The core task involves preprocessing audio data, extracting features, feature engineering, training models, evaluating their performance, and fine-tuning them. To achieve higher accuracy, we employed an ensemble method known as stacking to combine multiple models into a final classifier.
</p>
<p>
The development process began with data collection, initially utilizing the GTZAN dataset (1000 instances), which was subsequently expanded by crawling data from Spotify to create a more diverse dataset of 30,000 instances. This enhancement addressed the limitations of the small initial dataset, ensuring better model performance across various genres.
<p>
We deployed the model in multiple environments to maximize accessibility and usability. A local application was built using the `tkinter` library in Python, providing a user-friendly graphical interface for users to interact with the model on their local machines. Additionally, we created a Docker image to encapsulate all dependencies and configurations, ensuring consistent performance across different environments. The Dockerized application was then deployed using Fly.io, making it accessible via the internet with an improved user interface compared to the local application.
</p>
<p>
The model serves several purposes, enhancing both personal and commercial applications. For personal use, it automatically organizes music collections by genre, simplifies the creation of genre-specific playlists, and enhances the overall listening experience. We think it even can be used for commercial purpose: music streaming services can integrate the model to refine their recommendation algorithms, enabling genre-based search and discovery features, and personalizing user experiences.
</p>
<p>
The primary users of this model include music enthusiasts, developers integrating genre classification into music applications, and may be researchers studying music genres. This work demonstrates the potential of machine learning in automating and enhancing music genre classification, offering significant benefits to both individual users and commercial services.
</p>
Keywords: music genre classification, machine learning, feature engineering, ensemble method, data preprocessing, Docker, Fly.io, tkinter, recommendation systems, user experience.\

## Abbreviation 
// Ai thay gi chua co thi them vao 
## I. Introduction 
//Ha viet
## II. Data Preparation 
### 2.1 Data Collection <br>
// Khang viet
### 2.2 Data Preprocessing <br>
// Kien viet
### 2.3 Data Exploration <br>
// Nghia viet <br>
gom: <br>
- Descriptive Statistics: Summarizing the main characteristics of the data.
- Visualization: Creating plots and charts to visually inspect the data distributions, patterns, and anomalies.
## III. Modelling 
### 3.1 KNN
### 3.2 SVM
### 3.3 NEURAL-NET
### 3.4 ENSEMBLE: STACKING
## IV. Evaluation 
### 4.1 Metrics (Kien viet)
- Accuracy
- Time/Space taken
- Precision/Recall/F1
### 4.2 Validation/Tuning (Khang viet)
- Train-test-split
- Cross-validation -> Tuning using GridSearch
## V. Deployment <br>
// TA Viet
## VI. Further Enhancement <br>
// Ha viet
## Ref <br>
// Khang viet


