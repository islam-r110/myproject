# Real Estate Dashboard

This project is a Streamlit-based interactive dashboard designed for real estate company managers. It provides insights into real estate transactions in Dubai, allowing users to analyze sales data, identify trends, and make informed decisions.

## Project Structure

```
real-estate-dashboard
├── app.py                     # Main application file for the Streamlit dashboard
├── data                       # Directory containing the dataset
│   └── dubai_real_estate_data_realistic_500.csv  # Dataset with real estate transactions
├── requirements.txt           # List of dependencies for the project
└── README.md                  # Documentation for the project
```

## Features

- **Data Cleaning**: The application cleans the dataset by removing duplicates, handling missing values, and converting data types.
- **Interactive Visualizations**: Users can explore various metrics, including:
  - Top selling areas
  - Top buying nationalities
  - Best lead sources
  - Best property types
  - Top agents
  - Monthly sales trends

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd real-estate-dashboard
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Dashboard

To run the Streamlit dashboard, execute the following command in your terminal:
```
streamlit run app.py
```

Once the application is running, it will open in your default web browser, allowing you to interact with the dashboard.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.