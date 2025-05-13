
# Hotel Booking Status Predictor

This web application allows you to predict whether a hotel booking will be **Confirmed** or **Canceled** based on a set of input features. The app also integrates Power BI reports for visual analytics, providing a comprehensive solution for analyzing hotel booking data.

## Features:
- **Model Prediction**: Predicts booking status (Confirmed or Canceled) using pre-trained models (Random Forest, Extra Trees) or a custom model.
- **User Input**: Users can input booking details such as the number of adults, children, lead time, meal plan, room type, and more.
- **Power BI Integration**: Provides an option to embed a Power BI report via a URL input field, allowing users to visualize additional analytics.

## Technologies:
- Streamlit: Framework for building the web app.
- Scikit-learn: For machine learning models (Random Forest, Extra Trees).
- Pandas: For handling data and preprocessing.
- Joblib: For loading the pre-trained model.
- Requests: For downloading the model from a URL.
- Power BI: To embed external reports.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/hotel-booking-status-predictor.git
    cd hotel-booking-status-predictor
    ```

2. **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application**:
    ```bash
    streamlit run app.py
    ```

5. Open your browser and go to `http://localhost:8501` to interact with the application.

## How to Use:

1. **Choose or provide a model**:
    - You can either select one of the pre-trained models (Random Forest or Extra Trees) or upload your own `.pkl` model file via URL.

2. **Enter booking details**:
    - Fill out the form with booking details like number of adults, children, lead time, market segment, special requests, etc.

3. **Get the prediction**:
    - After entering the details, click on the "Predict Booking Status" button to get the prediction. The result will display whether the booking is predicted to be **Confirmed** or **Canceled**.

4. **Embed Power BI Report**:
    - Paste the URL of a Power BI report to embed it in the app. The Power BI report will be shown directly below the prediction section.

## Example Usage:

- **Pre-trained model**: Select "Random Forest" or "Extra Trees" from the dropdown and input booking details.
- **Custom model**: Upload your own `.pkl` model by pasting the URL in the text input field.
- **Power BI**: Paste the URL of a Power BI report to view it in the app.

## Requirements:
- Python 3.6+
- Streamlit
- Scikit-learn
- Pandas
- Joblib
- Requests

To install these requirements, run:
```bash
pip install streamlit scikit-learn pandas joblib requests
````

## Contributing:

Feel free to fork the repository, create a branch, and submit pull requests for any improvements or bug fixes.

## License:

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```

You can copy this Markdown directly to your GitHub repository's README file. It will render well on GitHub and provide all the necessary instructions for installation and usage.
```
