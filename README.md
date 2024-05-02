# Predictive Analytics for Stroke Prevention

## Project Overview
This project, developed by Team 4 for BC2407 Analytics 2 at Nanyang Technological University, focuses on developing an early detection model for stroke using machine learning techniques. By integrating this model into Singapore's healthcare system, our aim is to empower individuals and healthcare professionals with tools to predict and therefore prevent stroke occurrences, significantly reducing the stroke incidence rate in Singapore.



## Executive Summary
Stroke is a leading cause of death and disability in Singapore. This project utilizes a dataset to train various machine learning models including Logistic Regression, CART, Random Forest, and Neural Network to identify key predictors of stroke and develop an efficient predictive model.



## Business Problem and Opportunities
Singapore’s healthcare system faces challenges with proactive stroke prevention. Our project addresses this gap by integrating a predictive model into the public healthcare system, enhancing early detection and enabling preventative healthcare measures.



## Business Outcome Measures and Desired Targets
The initiative aims to integrate machine learning models into the healthcare system to provide early warnings and personalized health strategies, significantly reducing the incidence and impact of stroke in Singapore.



## Proposed Solutions and Methodology
The project involves data preprocessing, exploratory data analysis, and application of various predictive modeling techniques. We used SMOTE for addressing class imbalance and tested different models to ensure the highest accuracy and usability in real-world applications.

## Machine Learning Models and Analysis Results

### Models Employed
1. **Logistic Regression**
   - **Objective**: Predict the likelihood of stroke occurrence.
   - **Key Findings**: Certain expected predictors like BMI and smoking status were not statistically significant, suggesting potential variability in stroke risk factors.

2. **CART (Classification and Regression Trees)**
   - **Objective**: Classify stroke risk using a decision tree that splits on predictor variables.
   - **Key Findings**: Initially complex, the model was pruned for simplicity, yet maintained a relatively high false negative rate.

3. **Random Forest**
   - **Objective**: Improve prediction accuracy and robustness using an ensemble of decision trees.
   - **Key Findings**: High accuracy but with a high false negative rate, highlighting its robustness and complexity.

4. **Neural Network Model**
   - **Objective**: Capture complex, nonlinear interactions between variables.
   - **Key Findings**: Best performance in balancing accuracy with a low false negative rate, making it the preferred model for deployment.

### Comparative Analysis
- **Accuracy**: Random Forest achieved the highest accuracy (90.1%) but suffered from high false negative rates.
- **Selection Rationale**: The Neural Network was selected for its lower false negative rate, crucial for medical applications to minimize the risk of missing a stroke diagnosis.



### Implementation Recommendations
- **Healthcare System Integration**: The Neural Network model is recommended for integration into healthcare platforms to provide early warnings and personalized health recommendations.
- **Public Health Strategy**: This model can significantly contribute to preventive healthcare, ultimately reducing the stroke incidence and healthcare costs associated with stroke treatment.



## Business Implications
The integration of the chosen Neural Network model into Singapore’s healthcare system can transform public health strategies by:
- Providing early detection and personalized health interventions.
- Reducing the overall healthcare burden by preventing stroke occurrences.



## Conclusion
The use of advanced predictive analytics in healthcare represents a transformative approach to managing and preventing stroke, with the potential to significantly enhance patient outcomes and optimize healthcare resources.
