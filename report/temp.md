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
<p>
    Now we already have three different models: K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and a Neural Network (NN). Each of these models possesses unique characteristics that make them suitable for different types of data and learning tasks. KNN excels in capturing local patterns, SVM thrives in high-dimensional spaces, while NN can model complex, non-linear relationships. By combining these independent models, stacking leverages their complementary strengths to produce more robust predictions.
</p>
<p>
    By combining these models, stacking takes advantage of their diverse strengths. While KNN might excel in one part of the data, SVM might perform better in another, and NN might capture intricate patterns missed by the others. This diversity helps in covering different aspects of the data distribution, leading to a reduction in overall error. Individual models might suffer from high bias (underfitting) or high variance (overfitting). Stacking helps balance these issues by averaging out the errors of individual models. For instance, a high-bias model can be compensated by a low-bias but high-variance model, and vice versa..
</p>
Schematically, our stacking ensemble model look like this 
<br>

![stack1](https://github.com/ML-K67-HUST/MUSIC-GENRE-CLASSIFICATION_PROJ/assets/112315454/7f8ed030-770d-4fd4-b23e-24563a0133c6)

<br>
	  Properly combined models in a stacked ensemble tend to generalize better to unseen data. This is because the weaknesses of one model are mitigated by the strengths of another, leading to improved performance on test data. The first step involves training several base models (KNN, SVM, NN) on the training data. Each model will generate predictions on this data. These predictions are then used as inputs to a meta-model (also called a second-level model or combiner model, in our work we used Logistic Regression), which learns how to best combine these predictions to make the final decision.
</p>
<br>

![stack2](https://github.com/ML-K67-HUST/MUSIC-GENRE-CLASSIFICATION_PROJ/assets/112315454/21978e35-0eb1-4874-80f9-d85a7230c4c1)

<br>
<p>    
    To avoid overfitting, the training of the meta-model typically involves a cross-validation procedure (Instead of fitting the whole dataset for each base models). It split the training data into k-folds, train each base model on k-1 folds, and make predictions on the remaining fold. Collect the out-of-fold predictions for each base model. These predictions form a new dataset, which is then used to train the meta-model. When making predictions on new data, each base model makes a prediction, and these predictions are then used as inputs to the meta-model, which produces the final prediction.
</p>
<br>

![stack3](https://github.com/ML-K67-HUST/MUSIC-GENRE-CLASSIFICATION_PROJ/assets/112315454/e121f16a-1972-4c93-a7f7-0b454320347b)

<br>
<p>
  We set pass_through=True, now the final meta-model not only learns from the predictions of the base models but also has access to the original features. This can potentially improve the performance of the stacked ensemble model by providing more information to the final meta-model for making decisions.
</p>
<br>

![stack4](https://github.com/ML-K67-HUST/MUSIC-GENRE-CLASSIFICATION_PROJ/assets/112315454/a88a1ae2-8736-49df-9411-ac7cc62f3308)

<br>
<p>
	  By combining the strengths of each model,we expect stacking creates a robust predictive model that is typically more accurate than any of the individual models alone. 
</p>

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


