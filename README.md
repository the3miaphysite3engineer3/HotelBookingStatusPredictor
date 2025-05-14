
# Hotel Booking Status Predictor

This web application predicts whether a hotel booking will be **Confirmed** or **Canceled** based on user-provided input features. It integrates with Hugging Face for model loading, Gemini AI for cancellation prevention strategies, and Power BI for visual analytics, offering a robust solution for hotel booking analysis.

## Online URL: 
[Hotel Booking Status Predictor](https://hotelbookingstatuspredictor.streamlit.app/)

## Features:
- **Model Prediction**: Predicts booking status (Confirmed or Canceled) using a pre-trained Random Forest model or a custom model from Hugging Face.
- **User Input**: Allows users to input booking details, including number of adults, children, lead time, meal plan, room type, and more.
- **Gemini AI Integration**: Provides actionable strategies to prevent cancellations when a booking is predicted to be canceled.
- **Power BI Integration**: Embeds a Power BI report for visualizing booking data analytics.

## Technologies:
- **Streamlit**: Framework for building the web application.
- **Scikit-learn**: For the Random Forest machine learning model.
- **Pandas**: For data handling and preprocessing.
- **Joblib**: For loading pre-trained models.
- **Requests**: For downloading models from Hugging Face.
- **Google Generative AI**: For generating cancellation prevention strategies.
- **Power BI**: For embedding interactive data visualizations.

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

4. **Set environment variables**:
    - Set `HF_TOKEN` for Hugging Face API access.
    - Set `GEMINI_API_KEY` for Google Generative AI access.
    ```bash
    export HF_TOKEN='your-huggingface-token'
    export GEMINI_API_KEY='your-gemini-api-key'
    ```

5. **Run the application**:
    ```bash
    streamlit run app.py
    ```

6. Open your browser and go to `http://localhost:8501` to interact with the application.

## How to Use:

1. **Select or provide a model**:
   - Choose the pre-trained Random Forest model from Hugging Face or paste a custom Hugging Face model URL (e.g., `username/repo/blob/main/model.pkl`).

2. **Enter booking details**:
   - Input details such as number of adults, children, lead time, meal plan, room type, market segment, and special requests.

3. **Get the prediction**:
   - Click the "Predict Booking Status" button to see if the booking is predicted as **Confirmed** or **Canceled**.
   - If canceled, Gemini AI provides personalized strategies to prevent cancellation.

4. **View Power BI Report**:
   - The embedded Power BI report displays visual analytics of booking data below the prediction section.

## Example Usage:

- **Pre-trained model**: Select "Random Forest (Hugging Face)" and input booking details.
- **Custom model**: Paste a Hugging Face model URL in the text input field.
- **Gemini AI**: If a cancellation is predicted, view AI-generated strategies to reduce cancellation risk.
- **Power BI**: Analyze booking trends via the embedded report.

## Requirements:
- Python 3.6+
- Streamlit
- Scikit-learn
- Pandas
- Joblib
- Requests
- Google Generative AI

To install these requirements, run:
```bash
pip install streamlit scikit-learn pandas joblib requests google-generativeai
```

## Contributing:

Feel free to fork the repository, create a branch, and submit pull requests for improvements or bug fixes.

## License:

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact Information:

* **Mohammed Ahmed Mohammed Abdul-Aziz**
  Email: [mohammed@example.com](mailto:mohammed@example.com)

* **Nada Nasser Ragab**
  Email: [nadanasssrnasser309@gmail.com](mailto:nadanasssrnasser309@gmail.com)

* **Naira Mohamed Abdelbasset**
  Email: [eng.naira2311@gmail.com](mailto:eng.naira2311@gmail.com)

* **Abdallah Ahmed Mostafa**
  Email: [bedo.ahmedusa2001@gmail.com](mailto:bedo.ahmedusa2001@gmail.com)

* **George Joseph Basilious Tawadrous**
  Email: [georgejoseph5000@gmail.com](mailto:georgejoseph5000@gmail.com)

* **Saif El-Din Mohammad Moheb**
  Email: [seifmoh495@gmail.com](mailto:seifmoh495@gmail.com)

