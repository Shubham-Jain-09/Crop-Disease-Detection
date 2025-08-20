<h1 align="center" id="title">🌾 ArogyaKrishi 🌾 </h1>

<p align="center">
  <img src="https://github.com/user-attachments/assets/6b22e194-0e53-41a1-a33c-0a0b204e8999" alt="image">
</p>


<p align="center"><img src="https://socialify.git.ci/Blacksujit/ArogyaKrishi/image?forks=1&amp;issues=1&amp;language=1&amp;name=1&amp;owner=1&amp;pulls=1&amp;stargazers=1&amp;theme=Auto" alt="project-image"></p>

## 🌟This is our Project or prototype development to the SIH Hackathon 2024


## 🧾🧾 Table of Contents:

- [Problem Statement ID](#-problem-statement-id-sih---1638)
- [Our Approach](#-our-approach-)
- [Project Structure](#-project-structure)
- [System Architecture Design](#-system-architecture-design-)
- [Live Demos](#-live-demos-web-and-app)
- [Presentation](#-presentation)
- [Research and References](#-research-and-references-some-proven-theories-which-validate-this-poc)
- [Problem ArogyaKrishi Addresses and Solves](#️-problem-arogyakrishi-addresses-and-solves-)
- [Features](#-features-arogya-krishi-comes-with-)
- [Tech Stack Used](#-built-with)
- [Installation Steps](#️-installation-steps)
- [Feasibility Analysis](#-feasibility-analysis)
- [Potential Challenges & Risks](#-potential-challenges--risks)
- [Impact and Benefits](#-impact-and-benefits)
- [Contribution Guidelines](#-contribution-guidelines)
- [Contributors](#-contributors-team-anant)
- [License](#️-license)


###  👉 Problem statement ID: SIH - 1638 

Theme - Agri&tech 

Background: Crop diseases can devastate yields leading to significant financial losses for farmers. Early detection and timely intervention are crucial for effective management. Description: Develop an AI-driven system that analyzes crop images and environmental data to predict potential disease outbreaks. This system will provide farmers with actionable insights and treatment recommendations to mitigate risks. Expected Solution: A mobile and web-based application that utilizes machine learning algorithms to identify crop diseases and suggest preventive measures and treatments based on real-time data.


### 💡 Our Approach :

1.) Divided  the problem statement into chunks and pieces .

2.) begain the searching with the each segement in the research papaers , you can visit [kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) , and [UCI Machine learning Datasets](https://archive.ics.uci.edu/dataset/486/rice+leaf+diseases) , for the references and datasets,

3.) Then for the inspiration purpose we look for the several projects and repositories  available on the github and internet , you can visit , these repositories , [repo1](https://github.com/Blacksujit/Crop-Disease-Detection) , and 
 [repo2](https://github.com/Blacksujit/Green-Points) , [repo3
](https://github.com/Blacksujit/Harvestify).

4.) For the innovation and ideas , we have used [claude.ai](https://claude.ai/new) , [perlexity](https://www.perplexity.ai/) , [gpt](https://openai.com/index/gpt-4/) , [research papaers](https://paperswithcode.com/) 

5.) Determine what are the common Problem faced by the indian farmers and also researched about that for reference visit ,[ here](https://www.jiva.ag/blog/what-are-the-most-common-problems-and-challenges-that-farmers-face) , and how we can automate and mitigate it with the help of the Tech and innovation to an optimum level , such thtat the farmers life will be easier,

6.) After  we have then searched for some common agricultural practices followed and weather forecast analysis report in the past years, which will help us to analyse why the things have become  more complex for the crops to survive in the extreme conditions you can visit for the [data](https://www.nber.org/digest/202406/weather-forecasts-and-farming-practices-india)   

7.) After all the informationa has exhausted we have then move towards the most crucial part which is user behaviour , becuase it is the only thing where our product will ipmact and also , we were aware of how seamlessly we can onboard our user will create an , visibility and viability and easily accessibility of our product . so thouroughly researched about , how familier the indian farmer behaviour is  using tech , whether it is mobile application or it is and web application , we thaught every aspect in terms of the language barrier to UI complexity to the navigation e, every nity , greety is tackled so we couldn't miss out anything , for the same we have researched about the things [here](https://www.sciencedirect.com/science/article/abs/pii/S0959652624010795)

8.) At last we have iterated on our approach to look out if any breaches our setback is there or not  , if it is then we have to restart the things from scratch but there wasnt , after all these brainstorming with the team  and other things we were ready to move to the POC stage with  our idea.

9.) so thats all these was our approach for the problem statement .


### 📂 Project Structure:

```

Arogya_Krishi_MVP/
|
│
├── app.py                          # Main application file
├── requirements.txt                # List of dependencies
├── .env                            # Environment variables
│
├── models/                         # Directory for machine learning models
│   ├── RandomForest.pkl            # Crop recommendation model
│   ├── DenseNet121v2_95.h5        # Soil type prediction model
│   ├── SoilNet_93_86.h5           # SoilNet model
│   └── plant_disease_model.pth     # (if applicable) Disease prediction model
│
├── utils/                          # Utility functions and classes
│   ├── disease.py                  # Disease-related utilities
│   ├── fertilizer.py               # Fertilizer-related utilities
│   └── model.py                    # Model-related utilities (e.g., ResNet9)
│
├── uploads/                        # Directory for uploaded files
│   └── (user-uploaded images)      # Images uploaded by users
│
├── static/                         # Static files (CSS, JS, images)
│   ├── css/                        # CSS files
│   ├── js/                         # JavaScript files
│   └── images/                     # Images used in the application
│
│──── templates/                      # HTML templates for rendering
│    ├── index.html                  # Main page
│    ├── signup.html                 # Signup page
│    ├── login.html                  # Login page
│    ├── dashboard.html              # User dashboard
│    ├── crop.html                   # Crop recommendation page
│    ├── fertilizer.html              # Fertilizer suggestion page
│    ├── disease.html                # Disease prediction page
│    ├── disease-result.html         # Result page for disease prediction
│    ├── crop-result.html            # Result page for crop prediction
│    └── try_again.html              # Error handling page
│
│
│────.gitattributes
│
│────.gitignore
│
│────README.md
│
│────crop-disease-detection.ipynb
│
│────config.py
│
│
│────crop_prediction_based_on_numeric_value.ipynb
│
│
│────notebooks/
│        
│──── instance/         
│        │──── farmers_database.db
│
│
│-----images/
│
│──── data/
│          ...csv files
│
│──── Data_Preprocessed/
          .....cleaned_data
 
```

## 🧱 System Arcitecture Design :


![image](https://github.com/user-attachments/assets/8f8324f4-b3ef-4831-b0aa-475b18fc49ca)



## 🚀 Live Demos , Web and  App:

### 🕸️📲Web App:

<p>Experience ArogyaKrishi in action: <a href="https://youtu.be/yBajAQB9Kas?si=ilwix0wwiN533UYi" target="_blank">Watch the Demo   (Web APP Demo) </a></p>

### 📲 App Demo : (Under Developement Phase)

https://github.com/user-attachments/assets/129d96a3-861c-46a2-99bd-3d54791d74ac

## 👓 Presentation :

 [see PPT Presentation Here by TEAM ANANT](https://github.com/user-attachments/files/17748441/SIH_TEAM_ANANT_CROP_DISEASE-3.pdf)


## 📚📚 Research And References: (Some Proven Theories' which validates this POC)

1.	Development of Machine Learning Methods for Accurate Prediction of Plant Disease Resistance
(2024): https://www.sciencedirect.com/science/article/pii/S2095809924002431

2.	Chinese cabbage leaf disease prediction and classification using Naive Bayes VGG-19 convolution deep neural network (2024) : https://ieeexplore.ieee.org/document/10407076

3.	Image-based crop disease detection with federated learning (2023): https://www.nature.com/articles/ s41598-023-46218-5

4.	Deep learning-based crop disease prediction with web application (2023) : https://www.
sciencedirect.com/science/article/pii/S2666154323002715

5.	Seasonal Crops Disease Prediction and Classification Using Deep Convolutional Encoder Network (2019): https://link.springer.com/article/10.1007/s00034-019-01041-0
   Cropin app link: https://www.cropin.com/farming-   apps#:~:text=Cropin%20Grow%20is%20a%20robust, stakeholders%20in%20the%20agri%2Decosystem.



## ✅❇️  Problem  ArogyaKrishi Addresses and Solves :

**Arogya Krishi is an agricultural application that addresses critical challenges faced by farmers, including crop disease detection, optimal crop recommendations, and soil health assessment. It provides tailored fertilizer suggestions and integrates real-time weather data to help farmers make informed decisions. With a user-friendly interface and community engagement features, Arogya Krishi empowers farmers to enhance productivity and sustainability in their agricultural practices.**


### Arogya Krishi: Problem Descriptions and Solutions

Arogya Krishi is a comprehensive agricultural application designed to address various challenges faced by farmers and agricultural stakeholders. Below are the key problems it addresses and the solutions it provides:

**1. Crop Disease Detection:**

Problem: Farmers often struggle to identify diseases affecting their crops, leading to reduced yields and economic losses. Early detection is crucial for effective management and treatment.

Solution: Arogya Krishi utilizes advanced image classification techniques powered by machine learning to analyze images of crops. The application can accurately identify diseases and provide detailed information about the disease, enabling farmers to take timely action.

**2. Crop Recommendation:**

Problem: Farmers may lack knowledge about which crops are best suited for their specific soil types and climatic conditions, leading to poor crop choices and low productivity.
Solution: The application offers crop recommendation features based on soil analysis and environmental factors. By analyzing soil characteristics and local climate data, Arogya Krishi suggests optimal crops that can thrive in the given conditions, enhancing productivity and profitability.

**3.) Fertilizer Recommendations:**

Problem: Farmers often face challenges in determining the right type and amount of fertilizers to use, which can lead to overuse or underuse, affecting crop health and the environment.
Solution: The application generates tailored fertilizer recommendations based on the specific crop being cultivated, soil health, and environmental conditions. This helps farmers optimize their fertilizer usage, improving crop yields while minimizing environmental impact.

**4. Weather Data Integration:**

Problem: Farmers need access to real-time weather data to make informed decisions about planting, irrigation, and harvesting. Lack of timely weather information can lead to crop losses.
Solution: Arogya Krishi integrates weather data from reliable sources, providing farmers with current weather conditions, forecasts, and alerts. This information helps farmers plan their activities more effectively, reducing risks associated with adverse weather.

**6. User-Friendly Interface:**

Problem: Many agricultural technologies are complex and difficult for farmers to use, especially those with limited technical knowledge.
Solution: Arogya Krishi is designed with a user-friendly interface that simplifies navigation and usage. It provides clear instructions and visual aids, making it accessible to farmers of all backgrounds.

**7. Community and Knowledge Sharing:**

Problem: Farmers often work in isolation and may lack access to shared knowledge and experiences from their peers.
Solution: The application can facilitate community engagement by allowing users to share their experiences, tips, and best practices. This fosters a sense of community and encourages collaborative learning among farmers.

  
## 🧐 Features  (ArogyaKrishi Comes with) :

**Here're some of the project's best features:**

*   Multi Crop Disease Prediction .
*   Fertilizer Recommendation
*   Realtime Crop Disease Analysis Dashboard
*   Soil Type Classification
*   Farmer Data Tracking
*   Weather Analysis's and Report
*   Multilingual Support
*   Google IO translate in your language
*   Chat bot support assistance multilingual's
*   Farmer Management
*   Fertilizer Recommendations Based On Soil

## 💻 Tech Stack Used :

**Technologies used in the project:**

*   Flask
*   Model development Pipeline
*   Machinery Learning
*   Python
*   transformers
*   deep learning
*   neural networks
*   pretrained Models
*   HTML
*   CSS
*   Javascript
*   SCSS
*   SQL-alchemy


## 🛠️ Installation Steps:

<p>1. Clone the Project</p>

```
git clone https://github.com/Blacksujit/ArogyaKrishi.git
```

<p>2. Train The Models</p>

```
run crop_prediction_based_on_numerical_value.ipynb
```

<p>3. save the Pretrained Models</p>

```
model/
  resnet_mode_90.pkl
  ......pkl
   etc
```

<p>4. create the dotenv File at root of Project</p>

```
.env
```

<p>5. Add Your API Key</p>

```
OPEN_WEATHER_APIKEY=YOUR_WEATHER_API_KEY
HUGGINGFACE_LOGIN_TOKEN=YOUR_HUGGING_FACE_TOKEN

```

<p>6. Run the Project</p>

```
python app.py 
```

## 📈 Feasibility Analysis:

•	**High Feasibility:** Advanced ML models and cloud deployment enable real-time disease prediction.

•	**Scalability:** Supports multilingual features

•	**Personalized Alerts**: Farmers get alerts based on crop type, region, and disease severity.

## ⚓ Potential Challenges & Risks:

•	**Data Quality:** Poor data leads to inaccurate predictions

•	**Accuracy of AI Models:** Risk of false positives and false negatives cases.

•	**Environmental Variability:** Presence of Diverse conditions. Viability Analysis:

•	**Early Detection:** Detects diseases early, lowering treatment costs and preventing spread.

## 🪄🔮 Impact And Benifities:

•	**Economic Benefits:** Lowers disease management costs, increasing productivity.

•	**Environmental Impact:** Optimizes pesticide/fertilizer use, promoting sustainability.

•	**Data-Driven Decisions:** Provides real-time insights for efficient farm management.

•	**Resilience:** Enhances farming practices and reduces risks of disease outbreaks.

•	**Uniqueness:**

1. Identify crop diseases

2. Predict disease outbreaks

3. Recommend preventive measure

4. Optimize resource allocation 

5. Enhance agricultural productivity 

6. Real-time performance 

7. Userfriendliness 

8. Scalability


## 🍰 Contribution Guidelines:

We are Open For Contributions please  create an  seperate branch and make your chnages and raise and PR if it
matches the Project requirements and follow all the guidelines we will happy to merge it .

### ✨ Contributors: (TEAM ANANT)

1.) Abhishek Kute

2.) Sanskar Awati

3.) Manas Kotian

4.) Minal 

5.) Tanushri Kharkar


## 🛡️ License:

This project is licensed under the MIT.

 
