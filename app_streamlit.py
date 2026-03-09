import streamlit as st
import requests

# Title and description
st.title("⚡ Energy Consumption Forecast Web UI")
st.write(
    "This interface allows you to test the FastAPI backend. "
    "Enter the last 24 hours of energy consumption values to get a forecast."
)

# User input field
# Default example values are provided
values = st.text_area(
    "Enter the last 24 hourly values (comma-separated)",
    "0.1,0.2,0.3,0.4"
)

# Prediction button
if st.button("Predict"):
    try:
        # Convert the user input string into a list of floats
        # Example: '0.1,0.2,0.3' → [0.1, 0.2, 0.3]
        values_list = [float(v.strip()) for v in values.split(",")]

        # Prepare the payload for the API
        payload = {"values": values_list}

        # Send a POST request to the FastAPI endpoint
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)

        # If the API responds successfully, display the result
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Forecast result: {result['prediction']}")
            st.write("📊 Full JSON response:")
            st.json(result)
        else:
            # If the API returns an error, show details
            st.error(f"❌ API request failed! Status code: {response.status_code}")
            st.write(response.text)

    except ValueError:
        # Handle invalid input (non-numeric values)
        st.error("⚠️ Please enter only numeric values (e.g., 0.1, 0.2, 0.3).")

    except Exception as e:
        # Catch any other errors
        st.error(f"An unexpected error occurred: {e}")
