# Email Spam Classification System

This project involves building a machine learning model to classify emails as spam or not spam using a Support Vector Machine (SVM) and TF-IDF vectorization.

## Requirements

- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

## Dataset

The dataset used is `spam_ham_dataset.csv`, which contains email texts and labels indicating whether the email is spam (`1`) or not (`0`).

## Steps

1. **Load Dataset**: Load the dataset and display its info and label distribution.
2. **Data Preprocessing**: Split the dataset into training, validation, and test sets. Transform the email texts using TF-IDF vectorization.
3. **Model Training**: Use GridSearchCV to find the best hyperparameters for the SVM model. Train the model on the training set.
4. **Evaluation**: Evaluate the model on training, validation, and test sets. Calculate and print the accuracy and error rates.

## Usage

Run the script to execute the steps mentioned above. The script will output the training, validation, and test errors, along with the label distribution.

## Notes

- The data visualization part (bar plot) is commented out but can be uncommented for better insights.
- Classification reports are written to files but currently commented out.

## Results

- Train Error
- Validation Error
- Test Error

## Conclusion

This project demonstrates a basic email spam classification system using SVM and TF-IDF vectorization with Python.

---



![Slide2](https://github.com/user-attachments/assets/a5f26f7f-16f5-4be5-8b97-99ef4e616b9a)
![Slide4](https://github.com/user-attachments/assets/bcd0a31c-33b4-4012-b64d-7c085a9e33bc)
![Slide6](https://github.com/user-attachments/assets/10a1b4a2-bbbb-429f-a693-dde2b63e4386)
![Slide8](https://github.com/user-attachments/assets/74dbba1b-efc3-49a7-837f-ebaaf226764b)
![Slide10](https://github.com/user-attachments/assets/6c5d5c3e-9841-4c0d-bf25-943542c77cb3)
![Slide12](https://github.com/user-attachments/assets/23efd2d0-5034-49cf-9ab3-fb4cfdc718fa)
![Slide14](https://github.com/user-attachments/assets/af11f61c-d36a-42cb-a4ec-aaee410ec4d1)
