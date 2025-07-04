import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    mutual_info_score
)
import streamlit as st
import pickle
mi1=pd.read_csv("C:/Users/shanm/OneDrive/Desktop/project/voice_prediction/vocal_gender_features_new.csv")


with open("C:/Users/shanm/OneDrive/Desktop/project/voice_prediction/model1.pkl", 'rb') as file:
    model1 = pickle.load(file)

with open("C:/Users/shanm/OneDrive/Desktop/project/voice_prediction/sscaler.pkl", 'rb') as file:
    scaler = pickle.load(file)


#set home page with titles of other pages
st.sidebar.title('HOME')
page=st.sidebar.radio("Getpage",["Project Info","View Dataset","Exploratory Data Analysis","Final Classification model ","Final Clustering model",
                                 "Know the Unknown Voice",
                                    "Who Creates"])

if page=="Project Info":
    st.title("VOICE PREDICTION MODEL")
    #put one image 
    st.image("C:/Users/shanm/OneDrive/Desktop/voice_prediction.jpg")
    # give a intro to the website
    st.write("""In the evolving landscape of voice-based technologies, predictive modeling plays a vital role in extracting 
             actionable insights from audio signals. This project introduces a voice prediction model that leverages both 
             supervised learning algorithms and unsupervised clustering techniques to decode patterns within vocal data, 
             aiming to enhance applications like speaker profiling, voice segmentation, and emotion-aware systems.
 """)

    st.write("""By combining these approaches, the model not only achieves robust classification performance
              but also uncovers hidden structures and relationships within complex voice data. 
             This dual strategy strengthens the reliability of voice prediction pipelines and broadens their 
             potential use in smart assistants, call center analytics, and health diagnostics. """)

elif page=="View Dataset":
    st.header('VOICE PREDICTION DATASET')
    st.image("C:/Users/shanm/OneDrive/Desktop/voice_to_dataset.png")
    
    st.write(mi1)

elif page=="Exploratory Data Analysis":
    st.header("correlation Analysis")
    Query=st.selectbox("select below queries",["Correlation Matrix","What is the correlation between mean_spectral_centroid and std_spectral_centroid?",
                                         "What is the correlation between mean_spectral_centroid and mean_spectral_bandwidth?",
                                         "What is the correlation between mean_spectral_centroid, mean_spectral_contrast and spectral_kurtosis?",
                                         ])
    if Query=="Correlation Matrix":
        spearman_corr = mi1.corr(method='spearman')

        st.write("Spearman Correlation Matrix:")
        corr=pd.DataFrame(spearman_corr)
        st.write(corr)
        # Heatmap for visualization
        plt.figure(figsize=(24, 18))
        sns.heatmap(spearman_corr, annot=True, cmap='magma', vmin=-1, vmax=1)
        plt.title('Spearman Rank Correlation Matrix Heatmap')
        st.pyplot(plt)
  

    if Query == "What is the correlation between mean_spectral_centroid and std_spectral_centroid?":
        correlation_data = mi1[['mean_spectral_centroid', 'std_spectral_centroid']].corr()
        st.write(correlation_data)
        # Create a heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

        plt.title('Correlation Between mean_spectral_centroid and std_spectral_centroid')

        st.pyplot(plt)

    if Query == "What is the correlation between mean_spectral_centroid and mean_spectral_bandwidth?":
        correlation_data = mi1[['mean_spectral_centroid', 'mean_spectral_bandwidth']].corr()
        st.write(correlation_data)
        # Create a heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(correlation_data, annot=True, cmap='plasma', fmt=".2f", linewidths=0.5)

        plt.title('Correlation Between mean_spectral_centroid and mean_spectral_bandwidth')

        st.pyplot(plt)
    if Query ==  "What is the correlation between mean_spectral_centroid, mean_spectral_contrast and spectral_kurtosis?":   
        correlation_data = mi1[['mean_spectral_centroid', 'mean_spectral_contrast', 'spectral_kurtosis']].corr()
        st.write(correlation_data)

        # Create a heatmap
        selected_features = mi1[['mean_spectral_centroid', 'mean_spectral_contrast', 'spectral_kurtosis']]

        # Create the pair plot
        sns.pairplot(selected_features,  diag_kind='kde')

        plt.suptitle('Pair Plot of Spectral Audio Features', fontsize=14)
     
        
        st.pyplot(plt)
elif page=="Final Classification model ":
    st.header('XGBoost Model for Voice Prediction')
    st.image("C:/Users/shanm/OneDrive/Desktop/The-structure-of-XGB-model.png")
    #split the dataset
    X_train,X_test,y_train,y_test = train_test_split(mi1.drop(['label'],axis=1),mi1['label'])
    
    #normalize the values
    scaler = StandardScaler()

    X_train= scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)


    # Oversampling using SMOTE
    smt = SMOTE()
    X_train_smsampled, y_train_smsampled= smt.fit_resample(X_train, y_train)

#trains the xgboost model-final model
    model_xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=10,Randomstate=42)
    model_xgb.fit(X_train_smsampled, y_train_smsampled)

    # Print all model parameters
    print("Model Parameters:")
    print(model_xgb.get_params())

    # Predictions
    y_pred_train = model_xgb.predict(X_train_smsampled)
    y_pred_test = model_xgb.predict(X_test)

    # Compute evaluation metrics
    accuracy_train = accuracy_score(y_train_smsampled, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    precision_train = precision_score(y_train_smsampled, y_pred_train)
    precision_test = precision_score(y_test, y_pred_test)

    recall_train = recall_score(y_train_smsampled, y_pred_train)
    recall_test = recall_score(y_test, y_pred_test)

    f1_train = f1_score(y_train_smsampled, y_pred_train)
    f1_test = f1_score(y_test, y_pred_test)


    if st.button("Train Accuracy & Test Accuracy"):
        st.write(f"Train Accuracy: {accuracy_train}, Test Accuracy: {accuracy_test}")
    if st.button("Precision for Train & Test"):
        st.write(f"Train Precision: {precision_train},  Test Precision: {precision_test}")
    if st.button("Recall for Train & Test"):
        st.write(f"Train Recall: {recall_train}, Test Recall: {recall_test}")
    if st.button("F1 Score for Train & Test"):
        st.write(f"f1 Train: {f1_train},\t f1 Test : {f1_test}")

elif page=="Final Clustering model":
    st.header('K-means Clustering Model for Voice Prediction')
    dt = pd.read_csv("C:/Users/shanm/OneDrive/Desktop/project/voice_prediction/vocal_gender_features_new.csv")
    X1_train = dt.drop(['label'],axis=1)
    kmeans = KMeans(n_clusters=2)
    label1=kmeans.fit_predict(X1_train)

    # Unsupervised metrics
    silhouette = silhouette_score(X1_train, label1)
    db_index = davies_bouldin_score(X1_train, label1)
    ch_index = calinski_harabasz_score(X1_train, label1)


    #  Visualize the clusters
    plt.figure(figsize=(6, 4))
    plt.scatter(X1_train["mean_spectral_centroid"], X1_train["std_spectral_centroid"], c=label1, cmap='viridis', s=50)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                c='red', s=200, alpha=0.75, marker='X', label='Centroids')
    plt.title("K-Means Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    st.subheader(f"Silhouette Score: {silhouette:.3f}")
    st.subheader(f"Davies-Bouldin Index: {db_index:.3f}")
    st.subheader(f"Calinski-Harabasz Index: {ch_index:.3f}")
elif page=="Know the Unknown Voice":
    st.subheader("Give Details ")

    col1, col2 = st.columns(2)		
    with col1:
        mean_spectral_centroid=st.number_input("Enter mean_spectral_centroid", min_value=0.0, format="%.4f")
        std_spectral_centroid=st.number_input("Enter std_spectral_centroid ",  min_value=0.0, format="%.4f")
        mean_spectral_bandwidth=st.number_input("Enter your mean_spectral_bandwidth ",  min_value=0.0, format="%.4f")
        std_spectral_bandwidth=st.number_input("Enter std_spectral_bandwidth ",  min_value=0.0, format="%.4f")
        mean_spectral_contrast=st.number_input("Enter mean_spectral_contrast ",  min_value=0.0, format="%.4f")
        mean_spectral_flatness=st.number_input("Enter mean_spectral_flatness ",  min_value=0.0, format="%.4f")
        mean_spectral_rolloff=st.number_input("Enter mean_spectral_rolloff",  min_value=0.0, format="%.4f")
        zero_crossing_rate=st.number_input("Enter zero_crossing_rate",  min_value=0.0, format="%.4f")
        rms_energy=st.number_input("Enter rms_energy ",  min_value=0.0, format="%.4f")
        mean_pitch=st.number_input("Enter mean_pitch ",  min_value=0.0, format="%.4f")
        min_pitch=st.number_input("Enter min_pitch ",  min_value=0.0, format="%.4f")
        max_pitch=st.number_input("Enter max_pitch ",  min_value=0.0, format="%.4f")
        std_pitch=st.number_input("Enter your std_pitch",  min_value=0.0, format="%.4f")
        spectral_skew=st.number_input("Enter your spectral_skew",  min_value=0.0, format="%.4f")
        spectral_kurtosis=st.number_input("Enter spectral_kurtosis ",  min_value=0.0, format="%.4f")
        energy_entropy=st.number_input("Enter energy_entropy ",  min_value=0.0, format="%.4f")
        log_energy=st.number_input("Enter log_energy ",  min_value=0.0, format="%.4f")
        mfcc_1_mean=st.number_input("Enter mfcc_1_mean ",  min_value=0.0, format="%.4f")
        mfcc_1_std=st.number_input("Enter mfcc_1_std ",  min_value=0.0, format="%.4f")
        mfcc_2_mean=st.number_input("Enter mfcc_2_mean ",  min_value=0.0, format="%.4f")
        mfcc_2_std=st.number_input("Enter mfcc_2_std ",  min_value=0.0, format="%.4f")

    with col2:

        mfcc_3_mean=st.number_input("Enter mfcc_3_mean ",  min_value=0.0, format="%.4f")
        mfcc_3_std=st.number_input("Enter mfcc_3_std ",  min_value=0.0, format="%.4f")
        mfcc_4_mean=st.number_input("Enter mfcc_4_mean ",  min_value=0.0, format="%.4f")
        mfcc_4_std=st.number_input("Enter mfcc_4_std ",  min_value=0.0, format="%.4f")
        mfcc_5_mean=st.number_input("Enter mfcc_5_mean ",  min_value=0.0, format="%.4f")
        mfcc_5_std=st.number_input("Enter mfcc_5_std ",  min_value=0.0, format="%.4f")
        mfcc_6_mean=st.number_input("Enter mfcc_6_mean ",  min_value=0.0, format="%.4f")
        mfcc_6_std=st.number_input("Enter mfcc_6_std ",  min_value=0.0, format="%.4f")
        mfcc_7_mean=st.number_input("Enter mfcc_7_mean ",  min_value=0.0, format="%.4f")
        mfcc_7_std=st.number_input("Enter mfcc_7_std ",  min_value=0.0, format="%.4f")
        mfcc_8_mean=st.number_input("Enter mfcc_8_mean ",  min_value=0.0, format="%.4f")
        mfcc_8_std=st.number_input("Enter mfcc_8_std ",  min_value=0.0, format="%.4f")
        mfcc_9_mean=st.number_input("Enter mfcc_9_mean ",  min_value=0.0, format="%.4f")
        mfcc_9_std=st.number_input("Enter mfcc_9_std ",  min_value=0.0, format="%.4f")
        mfcc_10_mean=st.number_input("Enter mfcc_10_mean ",  min_value=0.0, format="%.4f")
        mfcc_10_std=st.number_input("Enter mfcc_10_std ",  min_value=0.0, format="%.4f")
        mfcc_11_mean=st.number_input("Enter mfcc_11_mean ",  min_value=0.0, format="%.4f")
        mfcc_11_std=st.number_input("Enter mfcc_11_std ",  min_value=0.0, format="%.4f")
        mfcc_12_mean=st.number_input("Enter mfcc_12_mean ",  min_value=0.0, format="%.4f")
        mfcc_12_std=st.number_input("Enter mfcc_12_std ",  min_value=0.0, format="%.4f")
        mfcc_13_mean=st.number_input("Enter mfcc_13_mean ",  min_value=0.0, format="%.4f")
        mfcc_13_std=st.number_input("Enter mfcc_13_std ",  min_value=0.0, format="%.4f")



    data=(mean_spectral_centroid, std_spectral_centroid,mean_spectral_bandwidth, std_spectral_bandwidth,
       mean_spectral_contrast, mean_spectral_flatness,
       mean_spectral_rolloff, zero_crossing_rate, rms_energy,
       mean_pitch, min_pitch, max_pitch, std_pitch,spectral_skew,
       spectral_kurtosis, energy_entropy, log_energy, mfcc_1_mean,
       mfcc_1_std, mfcc_2_mean, mfcc_2_std, mfcc_3_mean, mfcc_3_std,
       mfcc_4_mean, mfcc_4_std, mfcc_5_mean, mfcc_5_std, mfcc_6_mean,
       mfcc_6_std, mfcc_7_mean, mfcc_7_std, mfcc_8_mean, mfcc_8_std,
       mfcc_9_mean, mfcc_9_std, mfcc_10_mean, mfcc_10_std,
       mfcc_11_mean, mfcc_11_std, mfcc_12_mean, mfcc_12_std,
       mfcc_13_mean, mfcc_13_std)


    if data and st.button('PREDICT'):

        input_data = np.asarray(data).reshape(1, -1)
        input_data_scaled=scaler.transform(input_data)
        # Make predictions
        prediction = model1.predict(input_data_scaled)
   
        
        if prediction==1:
            st.subheader(" The predicted voice 'Male' voice")
        else:
            st.subheader("The voice is predicted voice 'Female' voice")


elif page=="Who Creates":
    col1, col2 = st.columns(2)			 																				

    with col1:
        st.image("https://thumbs.dreamstime.com/b/chibi-style-d-render-devops-engineer-playful-young-male-character-laptop-table-isolated-white-background-rendered-361524322.jpg")
    st.write("I am Shanmugasundaram, and this is my 2nd Machine Learning project after " \
        "joining the Data Science course on the Guvi platform. This project marks the beginning" \
        " of my journey into the world of data-driven decision-making and predictive analytics. " \
        "Through this project, I aim to apply the concepts learned in my course and build a model that provides " \
        "meaningful insights.")
    st.write("""Coming from an engineering background, I have always been intrigued by problem-solving,
                  automation, and analytical thinking. Machine Learning fascinates me as it combines mathematics, 
                 programming, and real-world applications to transform raw data into meaningful insights.""")