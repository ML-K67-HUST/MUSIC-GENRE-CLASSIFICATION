# Noi dung report o day
## Acknowledgement <br>
// Tuan Anh Viet
<p>
We would like to express my sincere gratitude to our Professor Khoat Than Quang for his invaluable guidance and support throughout the duration of this Music Genre Classification project. His expertise and encouragement were instrumental in the successful completion of the research.
</p>

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
<p>
The objective of this data collection effort was to gather music features from various genres available on Spotify. The data collected is intended for further analysis to understand musical attributes and their correlation with different genres. The data collection process was implemented using a Jupyter notebook, which facilitated the automated scraping and processing of music data from Spotify. There are four primary steps in the process:

- Setting Up the Environment: Configure Spotify API credentials to authenticate and fetch data.

- Defining the Playlist and Genre: Define the Spotify playlist link and the genre of music to be scraped. The user is required to input the playlist URL manually.

- Scraping Music Data.

- Extracting Music Features: After downloading the music tracks, the script processes the files to extract various audio features. This includes chroma, spectral contrast, and Mel-frequency cepstral coefficients (MFCCs).
</p>
<p>
The data collection process described above provides a systematic approach to gathering and analyzing musical features from Spotify playlists. This data serves as a foundation for in-depth analysis of musical characteristics across different genres.
</p>

### 2.2 Data Preprocessing <br>
- Nho noi ve train-test-split va stratify sampling
// Kien viet
### 2.3 Data Exploration <br>
// Nghia viet <br>
gom: <br>
- Descriptive Statistics: Summarizing the main characteristics of the data.
- Visualization: Creating plots and charts to visually inspect the data distributions, patterns, and anomalies.
## III. Modelling 
### 3.1 KNN
<p>
The first approach to our Music Genre Classification project is K-Nearest Neighbors (KNN), a non-parametric, supervised learning classifier which use proximity to make classifications about the grouping of an individual data point. Due to its simplicity and adaptibility, choosing KNN helps establishing a strong baseline for the understanding of music data and futre implementations of more sophisticated models. 
</p>
<p>
Each instance in the training dataset $D$ is represented by a vector in an n-dimensional space, e.g., $x_{i}=(x_{i1},x_{i2},...,x_{in})^{T}$ where each dimension represents an attribute. For a new instance $z$ added to the space, the algorithm computes the distance between each instance $x$ in $D$ and $z$, then determines a set $NB(z)$ of the nearest neighbors of $z$ and finally use the majority of the labels in $NB(z)$ to predict the label for $z$.
</p>
<p>
For the problem of Music Genre Classification, we evaluate the accuracy efficiency of KNN
algorithm in our dataset by considering a range of hyper-parameters confessed below:
	
- n_neighbors: This parameter defines how many "neighbor" data points should be considered in the process of major voting. In practice, the number of n_neighbors can vary, but it is encouraged to be greater than 1 to avoid noise or error in only one nearest neighbor and not too large to avoid over-generalization.
	
- $p$: This parameter decide which distance metric we are going to use. The overall distance
between two data points m and n in an a-dimensional space can be represented as: 
	$$d(\mathbf{m}, \mathbf{n}) = \sqrt[p]{\sum_{i=1}^{a} (m_i - n_i)^p}$$
If we set p = 1 and p = 2, we will acquire the Manhattan Distance and the Euclidean distance, respectively. Different distance metrics can affect the neighbor-choosing process.

- $weights$: This parameter determines the weight assigned to each data point considered. Typically, in the "majority vote," every data point is given equal importance. However, this can sometimes introduce biases, leading to inconsistent predictions. Weighted "votes" address this issue by adjusting the importance of each vote according to specific rules. This adjustment can significantly impact the model's output, either positively or negatively, depending on the nature of the data.
For this problem, we consider the following values for weight metric:
	- Uniform weights: All data points have the same weight. This is also the default setup
implemented by Scikit-learn.
	- Distance weights: Data points that are farther from the point being considered will have smaller weights. The weight formula $W$ based on the distance $d$ can be expressed as follows: $W=1/(d^2)$
 	This ensures that closer data points have a greater influence on the prediction.

The values of the above hyper-parameters considered in the implementation of this project are:
- n_neighbors: $[1,40]$;
- $p: [1,2]$;
- $w: ['uniform', 'distance']$; 

</p>

### 3.2 SVM
<p> 
The second approach to our Music Genre Classification project is Support Vector Machine (SVM), a supervised machine learning technique commonly known for classification purposes, and can also be utilized for regression tasks. SVM is especially proficient in dealing with datasets with a high number of dimensions and is recognized for its ability to handle situations where the number of dimensions surpasses the number of samples. The primary advantage of SVM is its ability to identify the optimal hyperplane that effectively divides the data into distinct classes.

- **Core Concept**: SVM helps identify the hyperplane that maximizes the margin, which is the distance between the hyperplane and the nearest data points from any class, referred to as support vectors.
  
- **Non-linearity Handling**: SVM uses kernel functions to transform data that is not linearly separable into a higher-dimensional space where a hyperplane can effectively distinguish between the classes. Popular kernel functions include.:
    - **Linear Kernel**: Suitable for linearly separable data.
      $$k\left( x, z \right) = x^T z$$
    - **Polynomial Kernel**: Useful for data that is not linearly separable but can be separated in a higher-dimensional space.
 	$$k\left( x, z \right) = \left(r + \gamma x^T z \right)$$
    - **Radial Basis Function (RBF) Kernel**: Effective in scenarios where the decision boundary is complex and non-linear.
 	$$k\left( x, z \right) = \exp\left(\gamma \||x-z\||^2 \right)$$
    - **Sigmoid Kernel**: Often used as an alternative to neural networks, the sigmoid kernel can introduce non-linearity similar to RBF. However, it may not perform as well in practice for high-dimensional spaces as RBF or polynomial kernels.
      	$$k\left( x, z \right) = \tanh\left(\gamma x^T z + r\right)$$
</p>

#### Key Hyperparameters

<p>
In SVM, important hyperparameters include the regularization parameter <strong>C</strong> and the parameters specific to the chosen kernel. The regularization parameter <strong>C</strong> plays a crucial role in balancing the desire to minimize training error with the need to avoid overfitting on the test data, ultimately impacting the margin of the classifier. 

- **Regularization Parameter C**: A smaller **C** value in an SVM model increases the margin between classes, allowing for some misclassifications but promoting overall generalization. On the other hand, a larger **C** value aims to accurately classify all training examples, potentially leading to overfitting as the model may become too complex and unable to generalize well to unseen data.

  ![Example on the effect of C parameter](https://github.com/ML-K67-HUST/MUSIC-GENRE-CLASSIFICATION_PROJ/blob/main/report/c%20effect.png)

- **Kernel Parameters**: For instance, the gamma value in the RBF and sigmoid kernels determines the extent to which a single training example impacts the model, while the degree parameter in the polynomial kernel specifies the complexity of the polynomial function used for classification. These parameters play a crucial role in fine-tuning the performance of the kernel methods.

The values of the above hyper-parameters considered in the implementation of this project are:
- **C**: [1, 300]
- Kernel: ['linear', 'poly', 'rbf', 'sigmoid']
- gamma: ['scale', 'auto']
  
</p>

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

![Screenshot from 2024-05-27 21-55-06](https://github.com/ML-K67-HUST/MUSIC-GENRE-CLASSIFICATION_PROJ/assets/112315454/7db50b70-3a20-4c3b-b597-fec12b0bedf0)


<br>
	  Properly combined models in a stacked ensemble tend to generalize better to unseen data. This is because the weaknesses of one model are mitigated by the strengths of another, leading to improved performance on test data. The first step involves training several base models (KNN, SVM, NN) on the training data. Each model will generate predictions on this data. These predictions are then used as inputs to a meta-model (also called a second-level model or combiner model, in our work we used Logistic Regression), which learns how to best combine these predictions to make the final decision.
</p>
<br>

![Screenshot from 2024-05-27 21-52-08](https://github.com/ML-K67-HUST/MUSIC-GENRE-CLASSIFICATION_PROJ/assets/112315454/d6897c33-efc9-42e9-b424-b24484863b82)

<br>
<p>    
    To avoid overfitting, the training of the meta-model typically involves a cross-validation procedure (Instead of fitting the whole dataset for each base models). It split the training data into k-folds, train each base model on k-1 folds, and make predictions on the remaining fold. Collect the out-of-fold predictions for each base model. These predictions form a new dataset, which is then used to train the meta-model. When making predictions on new data, each base model makes a prediction, and these predictions are then used as inputs to the meta-model, which produces the final prediction.
</p>
<br>

![stack3](https://github.com/ML-K67-HUST/MUSIC-GENRE-CLASSIFICATION_PROJ/assets/112315454/f603ce43-e78d-4f52-8c00-a34c92e4b21a)

<br>
<p>
  We set pass_through=True, now the final meta-model not only learns from the predictions of the base models but also has access to the original features. This can potentially improve the performance of the stacked ensemble model by providing more information to the final meta-model for making decisions.
</p>
<br>

![stack4](https://github.com/ML-K67-HUST/MUSIC-GENRE-CLASSIFICATION_PROJ/assets/112315454/b76cc780-8d03-4350-a9f0-95134a504065)


<br>
<p>
	  By combining the strengths of each model,we expect stacking creates a robust predictive model that is typically more accurate than any of the individual models alone. 
</p>

## IV. Evaluation 
### 4.1 Metrics (Khang viet)
- Accuracy
- Time/Space taken
- Precision/Recall/F1
### 4.2 Validation/Tuning (Khang viet)
- Cross-validation -> Tuning using GridSearch
## V. Deployment <br>
// TA Viet
## VI. Further Enhancement <br>
// Ha viet
## Ref <br>
// Khang viet
- Khoat Than. (2024, Semester 2). IT3190E - Machine Learning. Lecture presented at Hanoi University of Science and Technology, March 2024.
- Tiep, Vu Huu, "Machine Learning cơ bản", published March 27th, 2018.
- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition by Aurélien Géron - Released September, published on 2019, 39-129.
- Article Medium.

[1] Sturm, Bob L. "The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use." _arXiv preprint arXiv:1306.1461_ (2013).<br>
[2] Ghosh, Partha & Mahapatra, Soham & Jana, Subhadeep & Jha, Ritesh. (2023). A Study on Music Genre Classification using Machine Learning. International Journal of Engineering Business and Social Science. 1. 308-320. 10.58451/ijebss.v1i04.55.<br>
[3] A. Sandra Grace, "Song and Artist Attributes Analysis For Spotify," 2022 International Conference on Engineering and Emerging Technologies (ICEET), Kuala Lumpur, Malaysia, 2022, pp. 1-6, doi: 10.1109/ICEET56468.2022.10007360. keywords: {Deep learning;Data analysis;Terminology;Data visualization;Companies;Metadata;Media;Spotify;Spotify Web API;Attributes;Machine learning;Data visualization;Audio Features},<br>
[4] Andrana, "Work w/ Audio Data: Visualise, Classify, Recommend"  notebook on Kaggle <br>
[5] McFee, Brian & Raffel, Colin & Liang, Dawen & Ellis, Daniel & Mcvicar, Matt & Battenberg, Eric & Nieto, Oriol. (2015). librosa: Audio and Music Signal Analysis in Python. 18-24. 10.25080/Majora-7b98e3ed-003.<br>
[6] Lucas B.V. de Amorim, George D.C. Cavalcanti, Rafael M.O. Cruz. The choice of scaling technique matters for classification performance (23 Dec 2022)<br>
[7] Yu, Tong, and Hong Zhu. "Hyper-parameter optimization: A review of algorithms and applications." _arXiv preprint arXiv:2003.05689_ (2020).<br>
[8] Alexandropoulos, Stamatios-Aggelos & Aridas, Christos & Kotsiantis, Sotiris & Vrahatis, Michael. (2019). Stacking Strong Ensembles of Classifiers. 10.1007/978-3-030-19823-7_46. <br>
[9] Hossin, Mohammad & M.N, Sulaiman. (2015). A Review on Evaluation Metrics for Data Classification Evaluations. International Journal of Data Mining & Knowledge Management Process. 5. 01-11. 10.5121/ijdkp.2015.5201. <br>
[10] Bates, Stephen, Trevor Hastie, and Robert Tibshirani. "Cross-validation: what does it estimate and how well does it do it?." _Journal of the American Statistical Association_ (2023): 1-12.


